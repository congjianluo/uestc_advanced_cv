# Starter code prepared by James Hays for CS 143, Brown University

# This feature is inspired by the simple tiny images used as features in
#  80 million tiny images: a large dataset for non-parametric object and
#  scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
#  Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
#  pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

import numpy as np
import cv2


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def get_tiny_images(image_paths):
    image_feats = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (20, 20))
        image_feat = np.resize(image, [20 * 20])
        image_feat = image_feat.tolist()
        mean = np.mean(image_feat)
        image_feat = [(value - mean) for value in image_feat]
        # print(np.sum(image_feat))
        image_feats.append(image_feat)
    return image_feats
# image_paths is an N x 1 cell array of strings where each string is an
#  image path on the file system.
# image_feats is an N x d matrix of resized and then vectorized tiny
#  images. E.g. if the images are resized to 16x16, d would equal 256.

# To build a tiny image feature, simply resize the original image to a very
# small square resolution, e.g. 16x16. You can either resize the images to
# square while ignoring their aspect ratio or you can crop the center
# square portion out of each image. Making the tiny images zero mean and
# unit length (normalizing them) will increase performance modestly.

# suggested functions: imread, imresize
