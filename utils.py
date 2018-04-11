from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import math
import urllib
from keras.utils import to_categorical


def crop_2d(image, top_left_corner, height, width):
    """
    Returns a crop of an image.
    """

    x_start = top_left_corner[0]
    y_start = top_left_corner[1]
    x_end = x_start + width
    y_end = y_start + height

    return image[x_start:x_end, y_start:y_end, ...]


def prepare_images(image, size, ratio, n_slices):
    """
    Slices an image into chunks.
    """

    height = n_slices * ratio[1]
    width = n_slices * ratio[0]

    slice_height = int(size[1] / height)
    slice_width = int(size[0] / width)

    imgs = []

    for y in range(width):
        for x in range(height):
            tl_corner = (x * slice_height, y * slice_width)
            imgs.append(crop_2d(image, tl_corner, height=slice_height, width=slice_width))

    return np.stack(imgs)

def download_image(url, file_name):
    """
    Downloads an image from the internet.
    """
    return urllib.urlretrieve(url, file_name)
