import cv2
import numpy as np
from my_imfilter import my_imfilter
from utils import luo_imshow, luo_imwrite, luo_fspecial
from vis_hybrid_image import vis_hybrid_image

# this script has test cases to help you test my_imfilter() which you will
# write. You should verify that you get reasonable output here before using
# your filtering to construct a hybrid image in proj1.m. The outputs are
# all saved and you can include them in your writeup. You can add calls to
# imfilter() if you want to check that my_imfilter() is doing something
# similar.
cv2.destroyAllWindows()

## Setup
test_image = cv2.imread("../data/cat.bmp")
test_image = cv2.resize(test_image, (0, 0), fx=0.7, fy=0.7)


# luo_imwrite(test_image, "Figure 1")

## Identify filter
# This filter should do nothing regardless of the padding method you use.
def identity():
    identity_filter = np.array(([0, 0, 0], [0, 1, 0], [0, 0, 0]), dtype="float32")
    identity_image = my_imfilter(test_image, identity_filter)
    luo_imshow("Figure 2", identity_image)
    luo_imwrite(identity_image, 'identity_image.jpg')
    return identity_image


## Small blur with a box filter
# This filter should remove some high frequencies
def blur():
    blur_filter = np.array(([1, 1, 1], [1, 1, 1], [1, 1, 1]), dtype="float32")
    blur_filter = blur_filter / np.sum(blur_filter)
    blur_image = my_imfilter(test_image, blur_filter)
    luo_imshow("Figure 3", blur_image)
    luo_imwrite(blur_image, 'blur_image.jpg')
    return blur_image


## Large blur
# This blur would be slow to do directly, so we instead use the fact that
# #Gaussian blurs are separable and blur sequentially in each direction.

def large_1d_blur():
    large_1d_blur_filter = luo_fspecial(25, 1, 10)
    large_blur_image = my_imfilter(test_image, large_1d_blur_filter)
    large_blur_image = my_imfilter(large_blur_image, large_1d_blur_filter)
    luo_imshow("Figure 4", large_blur_image)
    luo_imwrite(large_blur_image, 'large_blur_image.jpg')
    return large_blur_image


# #If you want to see how slow this would be to do naively, try out this
# #equivalent operation:
# tic #tic and toc run a timer and then print the elapsted time
# large_blur_filter = fspecial('Gaussian', [25 25], 10);
# large_blur_image = my_imfilter(test_image, large_blur_filter);
# toc

## Oriented filter (Sobel Operator)

def sobel():
    sobel_filter = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]), dtype="float32")
    sobel_image = my_imfilter(test_image, sobel_filter)
    # 0.5 added because the output image is centered around zero otherwise and mostly black
    luo_imshow("Figure 5", sobel_image + 0.5)
    luo_imwrite(sobel_image + 0.5, "sobel_image.jpg")
    return sobel_image


## High pass filter (Discrete Laplacian)
def laplacian():
    laplacian_filter = np.array(([0, 1, 0], [1, -4, 1], [0, 1, 0]), dtype="float32")
    laplacian_image = my_imfilter(test_image, laplacian_filter)
    # 0.5 added because the output image is centered around zero otherwise and mostly black
    luo_imshow("Figure 6", laplacian_image + 0.5)
    luo_imwrite(laplacian_image + 0.5, "laplacian_image.jpg")
    return laplacian_image


## High pass "filter" alternative
def high_pass():
    high_pass_image = test_image - blur()
    luo_imshow("Figure 7", high_pass_image + 0.5)
    luo_imwrite(high_pass_image + 0.5, "high_pass_image.jpg")
    return high_pass_image


# 选择需要运行的test
if __name__ == "__main__":
    # identity()
    # blur()
    # large_1d_blur()
    # sobel()
    # laplacian()
    # high_pass()
    pass

# by Congjian Luo
