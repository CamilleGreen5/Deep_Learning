import tensorflow as tf
import os
import numpy as np
import json

from PIL import Image
from utils.box_utils import compute_target
from utils.image_utils import random_patching, horizontal_flip
from functools import partial

"""Different function to load Pascal Voc dataset, prepare images and generate batch from it"""


class COCODataset():
    """ Class for COCO Dataset

    Attributes:
        root_dir: dataset root dir (ex: ./data/VOCdevkit)
        num_examples: number of examples to be used
                      (in case one wants to overfit small data)
    """

    def __init__(self, root_dir, year, default_boxes,
                 new_size, liste_obj, num_examples=-1, clear_data=False, augmentation=None):
        super(COCODataset, self).__init__()
        self.idx_to_name = liste_obj
        self.name_to_idx = dict([(v, k)
                                 for k, v in enumerate(self.idx_to_name)])
        self.root_dir = root_dir
        self.image_dir = os.path.join(self.root_dir, 'Images')
        self.anno_dir = os.path.join(self.root_dir, 'Annotations')
        self.annotation_filename = os.path.join(self.anno_dir, 'instances_val2017.json')
        self.ids = list(map(lambda x: x[:-4], os.listdir(self.image_dir)))  # name de toutes les images
        self.default_boxes = default_boxes
        self.new_size = new_size

        list_pop = []
        # for index in range(len(self.ids)):
        #     img = self._get_image(index)
        #     np_img = np.array(img)
        #     s = np_img.shape
        #     if len(s) != 3:
        #         list_pop.append(index)
        #
        # if clear_data:
        #     for index in range(len(self.ids)):
        #         if index % 100 == 0:
        #             print(index)
        #         img = self._get_image(index)
        #         w, h = img.size
        #         _, labels = self._get_annotation(index, (h, w))
        #         nb = [0 for i in range(len(self.idx_to_name) + 1)]
        #         for label in labels:
        #             if 0 < label < len(self.idx_to_name) + 1:
        #                 nb[label] += 1
        #         if not (np.array(nb).any() != 0):
        #             list_pop.append(index)
        # for i in range(len(list_pop) - 1, 0, -1):
        #     try:
        #         # (self.ids).pop(list_pop[i])
        #         filename = str(self.ids[list_pop[i]]) + '.jpg'
        #         path_file = os.path.join(self.image_dir, filename)
        #         os.remove(path_file)
        #         print('x')
        #     except:
        #         print(path_file)
        #         print("file unfound")

        if num_examples != -1:
            self.ids = self.ids[:num_examples]

        self.train_ids = self.ids[:int(len(self.ids) * 0.75)]
        self.val_ids = self.ids[int(len(self.ids) * 0.75):]

        if augmentation is None:
            self.augmentation = ['original']
        else:
            self.augmentation = augmentation + ['original']

    def __len__(self):
        return len(self.ids)

    def _get_image(self, index):
        """ Method to read image from file
            then resize to (300, 300)
            then subtract by ImageNet's mean
            then convert to Tensor

        Args:
            index: the index to get filename from self.ids

        Returns:
            img: tensor of shape (3, 300, 300)
        """
        filename = self.ids[index]
        img_path = os.path.join(self.image_dir, filename + '.jpg')
        img = Image.open(img_path)

        return img

    def _get_annotation(self, index, orig_shape):
        """ Method to read annotation from file
            Boxes are normalized to image size
            Integer labels are increased by 1

        Args:
            index: the index to get filename from self.ids
            orig_shape: image's original shape

        Returns:
            boxes: numpy array of shape (num_gt, 4)
            labels: numpy array of shape (num_gt,)
        """
        h, w = orig_shape
        boxes = []
        labels = []
        image_id = 0

        with open(self.annotation_filename, 'r') as COCO:
            coco = json.loads(COCO.read())

        filename = self.ids[index]
        img_path = str(filename) + '.jpg'

        for img in coco['images']:
            if img['file_name'] == img_path:
                image_id = img['id']

        for obj in coco['annotations']:
            if obj['image_id'] == image_id:
                idx = obj['category_id']
                bbox = obj['bbox']
                xmin = (float(obj['bbox'][0]) - 1) / w
                ymin = (float(obj['bbox'][1]) - 1) / h
                xmax = (float(obj['bbox'][2]) - 1) / w
                ymax = (float(obj['bbox'][3]) - 1) / h
                boxes.append([xmin, ymin, xmin + xmax, ymin + ymax])
                nb = 0
                for categorie in self.idx_to_name:
                    id_c = self.name_to_idx[categorie] + 1
                    if id_c == idx:
                        nb += 1
                        labels.append(self.name_to_idx[categorie] + 1)
                if nb == 0:
                    labels.append(0)
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def generate(self, subset=None):
        """ The __getitem__ method
            so that the object can be iterable

        Args:
            index: the index to get filename from self.ids

        Returns:
            img: tensor of shape (300, 300, 3)
            boxes: tensor of shape (num_gt, 4)
            labels: tensor of shape (num_gt,)
        """
        if subset == 'train':
            indices = self.train_ids
        elif subset == 'val':
            indices = self.val_ids
        else:
            indices = self.ids
        for index in range(len(indices)):
            filename = indices[index]
            # img, orig_shape = self._get_image(index)
            img = self._get_image(index)
            w, h = img.size
            boxes, labels = self._get_annotation(index, (h, w))
            boxes = tf.constant(boxes, dtype=tf.float32)
            labels = tf.constant(labels, dtype=tf.int64)

            augmentation_method = np.random.choice(self.augmentation)
            if augmentation_method == 'patch':
                img, boxes, labels = random_patching(img, boxes, labels)
            elif augmentation_method == 'flip':
                img, boxes, labels = horizontal_flip(img, boxes, labels)

            img = np.array(img.resize(
                (self.new_size, self.new_size)), dtype=np.float32)
            img = (img / 127.0) - 1.0
            img = tf.constant(img, dtype=tf.float32)

            gt_confs, gt_locs = compute_target(
                self.default_boxes, boxes, labels)

            labels = labels.numpy()
            labels = labels.squeeze()
            sum_lab = 0
            for j in range(1, len(self.idx_to_name) + 1):
                lab_obj = labels[labels == j]
                sum_lab_obj = sum(lab_obj)
                sum_lab = sum_lab + sum_lab_obj

            yield filename, img, gt_confs, gt_locs, sum_lab


def create_batch_generator(root_dir, year, default_boxes,
                           new_size, batch_size, num_batches,
                           mode, liste_obj, clear_data,
                           augmentation=None):
    num_examples = batch_size * num_batches if num_batches > 0 else -1
    voc = COCODataset(root_dir, year, default_boxes,
                      new_size, liste_obj, num_examples, clear_data, augmentation)

    info = {
        'idx_to_name': voc.idx_to_name,
        'name_to_idx': voc.name_to_idx,
        'length': len(voc),
        'image_dir': voc.image_dir,
        'anno_dir': voc.anno_dir
    }

    if mode == 'train':  # separe en Train et Val dataset + shuffle et batch, renvoit tf dataset
        train_gen = partial(voc.generate, subset='train')
        train_dataset = tf.data.Dataset.from_generator(
            train_gen, (tf.string, tf.float32, tf.int64, tf.float32, tf.int64))
        val_gen = partial(voc.generate, subset='val')
        val_dataset = tf.data.Dataset.from_generator(
            val_gen, (tf.string, tf.float32, tf.int64, tf.float32, tf.int64))

        train_dataset = train_dataset.shuffle(40).batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

        return train_dataset.take(num_batches), val_dataset.take(-1), info
    else:
        dataset = tf.data.Dataset.from_generator(
            voc.generate, (tf.string, tf.float32, tf.int64, tf.float32, tf.int64))
        dataset = dataset.batch(batch_size)
        return dataset.take(num_batches), info
