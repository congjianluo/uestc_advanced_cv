import numpy as np
from utils import luo_fspecial
from scipy import ndimage, signal
import cv2
import math


def get_features(image, x, y, feature_width):
    dx = ndimage.sobel(image, 0)
    dy = ndimage.sobel(image, 1)

    features = np.zeros([len(x), 4, 4, 8])

    magnitude = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx) + math.pi
    angle = np.mod(np.floor(angle / (2 * math.pi) * 8), 8)
    angle = angle.astype(int)

    half_width = feature_width / 2

    for i in range(len(x)):
        px = x[i]
        py = y[i]

        x1 = max(px - half_width, 0)
        x2 = min(px + half_width - 1, image.shape[0])
        y1 = max(py - half_width, 0)
        y2 = min(py + half_width - 1, image.shape[1])

        for row in range(int(x1), int(x2)):
            for col in range(int(y1), int(y2)):
                if col >= 768:
                    print()
                cell_row = np.mod(math.floor((row - x1) / (feature_width / 4)), 4)
                cell_col = np.mod(math.floor((col - y1) / (feature_width / 4)), 4)
                features[i, cell_col, cell_row, angle[col, row]] = \
                    features[i, cell_col, cell_row, angle[col, row]] \
                    + magnitude[col, row]

    features = np.resize(features, [features.shape[0], 128])
    for i in range(features.shape[0]):
        t = max(np.amax(np.sqrt(np.multiply(features[i], features[i]))),1)
        features[i] = features[i] / t
    return features
