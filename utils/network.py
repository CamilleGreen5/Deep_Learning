from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
import os
from utils.layers import create_vgg16_layers, create_extra_layers, create_conf_head_layers, create_loc_head_layers
from utils.anchor import generate_default_boxes
from utils.box_utils import decode, compute_nms
from utils.losses import create_losses
import yaml
import time

""" Different function to create the custom keras model class: ssd model """


class SSD(Model):
    """ Class for SSD model
    Attributes:
        num_classes: number of classes
    """

    def __init__(self, num_classes):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.image_size = (300, 300)
        self.vgg16_conv4, self.vgg16_conv7 = create_vgg16_layers()  # initialise layers du feature extractor
        self.batch_norm = layers.BatchNormalization(  # rencentre et normalise variance input
            beta_initializer='glorot_uniform',
            gamma_initializer='glorot_uniform'
        )
        self.extra_layers = create_extra_layers()  # crée les layers du ssd
        self.conf_head_layers = create_conf_head_layers(num_classes)  # Create layers for classification
        self.loc_head_layers = create_loc_head_layers()  # Create layers for classification

        with open('data/config.yml') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        try:
            self.config = cfg["SSD300"]
        except AttributeError:
            raise ValueError('Unknown architecture')

    def compute_heads(self, x, idx):
        """ Compute outputs of classification and regression heads
        Args:
            x: the input feature map
            idx: index of the head layer
        Returns:
            conf: output of the idx-th classification head
            loc: output of the idx-th regression head
        """
        conf = self.conf_head_layers[idx](x)
        loc = self.loc_head_layers[idx](x)
        try:
            conf = tf.reshape(conf, [conf.shape[0], -1, self.num_classes])
            loc = tf.reshape(loc, [loc.shape[0], -1, 4])
        except:
            conf = tf.reshape(conf, [1, -1, self.num_classes])
            loc = tf.reshape(loc, [1, -1, 4])

        return conf, loc

    def init_vgg16(self):
        """ Initialize the VGG16 layers from pretrained weights
            and the rest from scratch using xavier initializer
        """
        origin_vgg = VGG16(weights='imagenet')
        for i in range(len(self.vgg16_conv4.layers)):
            self.vgg16_conv4.get_layer(index=i).set_weights(
                origin_vgg.get_layer(index=i).get_weights())

        fc1_weights, fc1_biases = origin_vgg.get_layer(index=-3).get_weights()
        fc2_weights, fc2_biases = origin_vgg.get_layer(index=-2).get_weights()

        conv6_weights = np.random.choice(
            np.reshape(fc1_weights, (-1,)), (3, 3, 512, 1024))
        conv6_biases = np.random.choice(
            fc1_biases, (1024,))

        conv7_weights = np.random.choice(
            np.reshape(fc2_weights, (-1,)), (1, 1, 1024, 1024))
        conv7_biases = np.random.choice(
            fc2_biases, (1024,))

        self.vgg16_conv7.get_layer(index=2).set_weights(
            [conv6_weights, conv6_biases])
        self.vgg16_conv7.get_layer(index=3).set_weights(
            [conv7_weights, conv7_biases])

    def call(self, x, **kwargs):
        """ The forward pass
        Args:
            x: the input image
        Returns:
            confs: list of outputs of all classification heads
            locs: list of outputs of all regression heads
        """
        confs = []
        locs = []
        head_idx = 0
        for i in range(len(self.vgg16_conv4.layers)):
            # print(self.vgg16_conv4.get_layer(index=i).name)
            x = self.vgg16_conv4.get_layer(index=i)(x)

            if i == len(self.vgg16_conv4.layers) - 5:
                conf, loc = self.compute_heads(self.batch_norm(x), head_idx)
                confs.append(conf)
                locs.append(loc)
                head_idx += 1

        x = self.vgg16_conv7(x)

        conf, loc = self.compute_heads(x, head_idx)

        confs.append(conf)
        locs.append(loc)
        head_idx += 1

        for layer in self.extra_layers:
            x = layer(x)
            conf, loc = self.compute_heads(x, head_idx)
            confs.append(conf)
            locs.append(loc)
            head_idx += 1

        confs = tf.concat(confs, axis=1)
        locs = tf.concat(locs, axis=1)

        return confs, locs

    def predict(self, imgs, **kwargs):

        default_boxes = generate_default_boxes(self.config)

        confs, locs = self.call(imgs)

        confs = tf.squeeze(confs, 0)
        locs = tf.squeeze(locs, 0)

        confs = tf.math.softmax(confs, axis=-1)
        # classes = tf.math.argmax(confs, axis=-1)
        # scores = tf.math.reduce_max(confs, axis=-1)

        boxes = decode(default_boxes, locs)

        out_boxes = []
        out_labels = []
        out_scores = []

        for c in range(1, self.num_classes):
            cls_scores = confs[:, c]
            score_idx = cls_scores > 0.6
            # cls_boxes = tf.boolean_mask(boxes, score_idx)
            # cls_scores = tf.boolean_mask(cls_scores, score_idx)
            cls_boxes = boxes[score_idx]
            cls_scores = cls_scores[score_idx]

            nms_idx = compute_nms(cls_boxes, cls_scores, 0.45, 200)
            cls_boxes = tf.gather(cls_boxes, nms_idx)
            cls_scores = tf.gather(cls_scores, nms_idx)
            cls_labels = [c] * cls_boxes.shape[0]

            out_boxes.append(cls_boxes)
            out_labels.extend(cls_labels)
            out_scores.append(cls_scores)

        out_boxes = tf.concat(out_boxes, axis=0)
        out_scores = tf.concat(out_scores, axis=0)

        boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
        classes = np.array(out_labels)
        scores = out_scores.numpy()

        return boxes, classes, scores

    @tf.function
    def train_step(self, imgs, gt_confs, gt_locs, criterion, optimizer, weight_decay):
        with tf.GradientTape() as tape:  # pour calculer le gradient ensuite (backpropagation)
            confs, locs = self(imgs)

            conf_loss, loc_loss = criterion(
                confs, locs, gt_confs, gt_locs)

            loss = conf_loss + loc_loss
            l2_loss = [tf.nn.l2_loss(t) for t in self.trainable_variables]
            l2_loss = weight_decay * tf.math.reduce_sum(l2_loss)
            loss += l2_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        print('.')

        return loss, conf_loss, loc_loss, l2_loss

    def train(self, batch_generator, val_generator, neg_ratio, num_epoch, weight_decay, steps_per_epoch, optimizer,
              train_summary_writer, val_summary_writer, checkpoint_dir):

        criterion = create_losses(neg_ratio, self.num_classes)

        for epoch in range(num_epoch):

            avg_loss = 0.0
            avg_conf_loss = 0.0
            avg_loc_loss = 0.0
            start = time.time()

            for i, (_, imgs, gt_confs, gt_locs, sum_lab) in enumerate(batch_generator):

                loss, conf_loss, loc_loss, l2_loss = self.train_step(imgs, gt_confs, gt_locs, criterion,
                                                                     optimizer, weight_decay)
                avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)
                avg_conf_loss = (avg_conf_loss * i + conf_loss.numpy()) / (i + 1)
                avg_loc_loss = (avg_loc_loss * i + loc_loss.numpy()) / (i + 1)

                if i % 10 == 0:
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', avg_loss, step=int(steps_per_epoch * epoch + i))
                        tf.summary.scalar('conf_loss', avg_conf_loss, step=int(steps_per_epoch * epoch + i))
                        tf.summary.scalar('loc_loss', avg_loc_loss, step=int(steps_per_epoch * epoch + i))

                    print('Epoch: {} Batch {} Time: {:.2}s | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f}'.format(
                        epoch + 1, i + 1, time.time() - start, avg_loss, avg_conf_loss, avg_loc_loss))

            avg_val_loss = 0.0
            avg_val_conf_loss = 0.0
            avg_val_loc_loss = 0.0

            for i, (_, imgs, gt_confs, gt_locs, sum_lab) in enumerate(val_generator):

                val_confs, val_locs = self(imgs)
                val_conf_loss, val_loc_loss = criterion(val_confs, val_locs, gt_confs, gt_locs)
                val_loss = val_conf_loss + val_loc_loss
                avg_val_loss = (avg_val_loss * i + val_loss.numpy()) / (i + 1)
                avg_val_conf_loss = (avg_val_conf_loss * i + val_conf_loss.numpy()) / (i + 1)
                avg_val_loc_loss = (avg_val_loc_loss * i + val_loc_loss.numpy()) / (i + 1)

                if i % 10 == 0:
                    with val_summary_writer.as_default():
                        tf.summary.scalar('loss', avg_val_loss, step=int(steps_per_epoch * epoch + i))
                        tf.summary.scalar('conf_loss', avg_val_conf_loss, step=int(steps_per_epoch * epoch + i))
                        tf.summary.scalar('loc_loss', avg_val_loc_loss, step=int(steps_per_epoch * epoch + i))

            self.save_weights(
                os.path.join(checkpoint_dir, 'ssd_epoch_{}.h5'.format(epoch + 1)))


def create_ssd(num_classes, pretrained_type, checkpoint_path):
    """ Create SSD model and load pretrained weights
    Args:
        num_classes: number of classes
        pretrained_type: type of pretrained weights, can be either 'VGG16' or 'ssd'
    Returns:
        net: the SSD model
    """

    net = SSD(num_classes)
    net(tf.random.normal((1, 300, 300, 3)))

    if pretrained_type == 'base':
        net.init_vgg16()

    elif pretrained_type == 'specified':
        if not os.path.isfile(checkpoint_path):
            raise ValueError(
                'Not a valid checkpoint file: {}'.format(checkpoint_path))
        try:
            net.load_weights(checkpoint_path)
        except Exception as e:
            raise ValueError(
                'Please check the following\n1./ Is the path correct ?\n2./ Is the model architecture correct ?')
    else:
        raise ValueError('Unknown pretrained type: {}'.format(pretrained_type))
    return net
