"""
Script to test detection on images (not annoted)
test images have to be in the --data-dir folder and to be in RGB
"""

import sys
import os
import cv2 as cv
import argparse
from utils.network import create_ssd
from utils.image_process import process_images, draw_to_image


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='data/examples_for_detection')
parser.add_argument('--num-examples', default=-1, type=int)
parser.add_argument('--checkpoint-path', default='./data/checkpoints/ssd_human_epoch_30.h5')
parser.add_argument('--gpu-id', default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


if __name__ == '__main__':

    f = open('data/custom_classes.txt', 'r')
    list_classes = f.readlines()
    list_classes.pop(0)
    NUM_CLASSES = len(list_classes) + 1

    list_image_name = os.listdir(args.data_dir)
    # print(liste_image_name)

    try:
        ssd = create_ssd(NUM_CLASSES, 'specified', args.checkpoint_path)
    except Exception as e:
        print(e)
        print('The program is exiting...')
        sys.exit()

    os.makedirs('data/detection_outputs', exist_ok=True)

    for image_name in list_image_name:

        image_path = os.path.join(args.data_dir, image_name)

        original_image, imgs = process_images(image_path, ssd.image_size)
        size = original_image.size

        boxes, classes, scores = ssd.predict(imgs)

        print('score = ', scores)
        print('classes = ', classes)
        print('boxes = ', boxes)

        original_image = draw_to_image(original_image, ssd, boxes, classes)

        cv.imwrite("data/detection_outputs/detected" + image_name, original_image)
