
import argparse
import tensorflow as tf
import os
import sys

from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from utils import csv_data, coco_data, voc_data
from utils.network import create_ssd


"""
Script to train the model
Data have to be : Pascal Voc dataset or Coco dataset(cleared from greyscale images)
A good idea could be to clear data by setting --clear-data to True the fist time (it will keep only images where there 
is classes you want)
Logs are saved every 10 batches
"""


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='voc')
parser.add_argument('--data-dir', default='../../Documents')   #/COCO
parser.add_argument('--batch-size', default=6, type=int)
parser.add_argument('--num-batches', default=2, type=int)
parser.add_argument('--neg-ratio', default=3, type=int)
parser.add_argument('--initial-lr', default=1e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=5e-4, type=float)
parser.add_argument('--num-epochs', default=1, type=int)
parser.add_argument('--checkpoint-dir', default='data/checkpoints/')
parser.add_argument('--checkpoint-name', default='ssd_human_epoch_30.h5')
parser.add_argument('--pretrained-type', default='specified')
parser.add_argument('--gpu-id', default='0')
parser.add_argument('--clear-data', default=False, type=bool)
parser.add_argument('--logs-dir', default="logs")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

checkpoint_path = args.checkpoint_dir + args.checkpoint_name

f = open('data/custom_classes.txt', 'r')
liste_obj = f.readlines()
liste_obj.pop(0)

NUM_CLASSES = len(liste_obj) + 1


if __name__ == '__main__':

    os.makedirs(args.logs_dir, exist_ok=True)     # crée dossier logs
    os.makedirs(args.checkpoint_dir, exist_ok=True)     # crée dossier logs
    os.makedirs(args.logs_dir + "/train", exist_ok=True)     # crée dossier logs
    os.makedirs(args.logs_dir + "/val", exist_ok=True)     # crée dossier logs

    try:
        ssd = create_ssd(NUM_CLASSES, args.pretrained_type, checkpoint_path)
    except Exception as e:
        print(e)
        print('The program is exiting...')
        sys.exit()

    if args.dataset == 'voc':
        batch_generator, val_generator, info = voc_data.create_batch_generator(args.data_dir, ssd, args.batch_size,
                                        args.num_batches, 'train', liste_obj, args.clear_data, augmentation=['flip'])

    # elif args.dataset == 'coco':
    #     batch_generator, val_generator, info = coco_data.create_batch_generator(      # crée les batch (resize image aussi)
    #         args.data_dir, args.data_year, ssd,
    #         args.batch_size, args.num_batches,
    #         'train', liste_obj, args.clear_data, augmentation=['flip'])  # the patching algorithm is currently causing bottleneck sometimes

    print('\ninfo : ', info)

    steps_per_epoch = info['length'] // args.batch_size

    lr_fn = PiecewiseConstantDecay(
        boundaries=[int(steps_per_epoch * args.num_epochs * 2 / 3),
                    int(steps_per_epoch * args.num_epochs * 5 / 6)],
        values=[args.initial_lr, args.initial_lr * 0.1, args.initial_lr * 0.01])

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_fn,
        momentum=args.momentum)

    train_summary_writer = tf.summary.create_file_writer(args.logs_dir + "/train")
    val_summary_writer = tf.summary.create_file_writer(args.logs_dir + "/val")

    try:
        ssd.train(batch_generator, val_generator, args.neg_ratio, args.num_epochs, args.weight_decay, steps_per_epoch,
                                        optimizer, train_summary_writer, val_summary_writer, args.checkpoint_dir)
    except Exception as e:
        print(e)
        print('\n ERROR IN TRAINING : The program is exiting...')
        sys.exit()

    print('end of training')