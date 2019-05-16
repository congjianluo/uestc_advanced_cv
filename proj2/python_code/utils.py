import cv2
import os
import numpy as np


def luo_imshow(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)


def luo_imwrite(image, name):
    path = os.path.join("../experience_results", name)
    cv2.imwrite(path, image)


def luo_fspecial(r, c, sigma):
    # MATLAB
    # H = fspecial('Gaussian', [r, c], sigma);
    return np.multiply(cv2.getGaussianKernel(r, sigma), (cv2.getGaussianKernel(c, sigma)).T)
