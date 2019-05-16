# Returns a set of interest points for the input image

# 'image' can be grayscale or color, your choice.
# 'feature_width', in pixels, is the local feature width. It might be
#   useful in this function in order to (a) suppress boundary interest
#   points (where a feature wouldn't fit entirely in the image, anyway)
#   or(b) scale the image filters being used. Or you can ignore it.

# 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
# 'confidence' is an nx1 vector indicating the strength of the interest
#   point. You might use this later or not.
# 'scale' and 'orientation' are nx1 vectors indicating the scale and
#   orientation of each interest point. These are OPTIONAL. By default you
#   do not need to make scale and orientation invariant local features.

import numpy as np
import random
import cv2
from scipy import ndimage, signal

from utils import luo_fspecial


def get_interest_points(image, feature_width):
    alpha = 0.04
    ix = ndimage.sobel(image, 0)
    iy = ndimage.sobel(image, 1)
    ix2 = ix * ix
    iy2 = iy * iy
    ixy = ix * iy
    ix2 = ndimage.gaussian_filter(ix2, sigma=2)
    iy2 = ndimage.gaussian_filter(iy2, sigma=2)
    ixy = ndimage.gaussian_filter(ixy, sigma=2)
    c, l = image.shape
    result = np.zeros((c, l))
    har = np.zeros((c, l))
    har_max = 0

    print('looking for corner . . .')
    for i in range(c):
        for j in range(l):
            # print('test ', j)
            m = np.array([[ix2[i, j], ixy[i, j]], [ixy[i, j], iy2[i, j]]], dtype=np.float64)
            har[i, j] = np.linalg.det(m) - alpha * (np.power(np.trace(m), 2))
            if har[i, j] > har_max:
                har_max = har[i, j]

    threshold = 0.01

    print('threshold')
    for i in range(c - 1):
        for j in range(l - 1):
            if har[i, j] > threshold * har_max and har[i, j] > har[i - 1, j - 1] and har[i, j] > har[i - 1, j + 1] \
                    and har[i, j] > har[i + 1, j - 1] and har[i, j] > har[i + 1, j + 1]:
                result[i, j] = 1

    x, y = np.where(result == 1)
    # confidence = [0] * har_max
    return x, y
