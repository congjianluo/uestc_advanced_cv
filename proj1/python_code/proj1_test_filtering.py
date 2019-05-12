import cv2
import numpy as np
from my_imfilter import my_imfilter
from utils import luo_imshow, luo_imwrite
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
identity_filter = np.array(([0, 0, 0], [0, 1, 0], [0, 0, 0]), dtype="float32")
identity_image = my_imfilter(test_image, identity_filter)
# luo_imshow("Figure 2", identity_image)
luo_imwrite(identity_image, 'identity_image.jpg')

## Small blur with a box filter
# This filter should remove some high frequencies
blur_filter = np.array(([1, 1, 1], [1, 1, 1], [1, 1, 1]), dtype="float32")
blur_filter = blur_filter / np.sum(blur_filter)
blur_image = my_imfilter(test_image, blur_filter)
# luo_imshow("Figure 3", blur_image)
luo_imwrite(identity_image, 'blur_image.jpg')

## Large blur
# This blur would be slow to do directly, so we instead use the fact that
# %Gaussian blurs are separable and blur sequentially in each direction.
