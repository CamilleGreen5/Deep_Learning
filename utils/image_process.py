import numpy as np
import tensorflow as tf
import cv2 as cv
from PIL import Image


def process_images(image_path, new_size):

    original_image = Image.open(image_path)
    image = np.array(original_image.resize(new_size), dtype=np.float32)
    image = (image / 127.0) - 1.0
    image = np.expand_dims(image, axis=0)
    imgs = tf.constant(image, dtype=tf.float32)

    return original_image, imgs


def draw_to_image(original_image, ssd, boxes, classes):

    original_image = np.array(original_image.resize(ssd.image_size))
    original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
    boxes *= ssd.image_size * 2
    if len(classes) > 0:
        color = [255, 0, 255]
        for i in range(len(classes)):
            x_min, y_min, x_max, y_max = map(int, boxes[i])
            cv.rectangle(original_image, (x_min, y_min), (x_max, y_max), color=color)

    return original_image
