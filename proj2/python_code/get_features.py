import numpy as np
from utils import luo_fspecial
from scipy import ndimage, signal
import cv2
import math


def get_features(image, x, y, feature_width):
    num_points = len(x)
    features = np.zeros(len(x), 128)

    dx = ndimage.sobel(image, 0)
    dy = ndimage.sobel(image, 1)

    magnitude = math.sqrt(dx * dx + dy * dy)
    angle = math.atan2(dy, dx) + math.pi
    angle = math.mod(math.floor(angle / (2 * math.pi) * 8), 8) + 1

    half_width = feature_width / 2;
