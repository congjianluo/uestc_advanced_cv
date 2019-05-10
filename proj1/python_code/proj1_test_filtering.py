import cv2
import numpy as np
from my_imfilter import my_imfilter
from vis_hybrid_image import vis_hybrid_image

test_image = cv2.imread("../data/cat.bmp")

test_image = cv2.resize(test_image, (0, 0), fx=0.5, fy=0.5)

identity_filter = np.array(([0, 0, 0], [0, 1, 0], [0, 0, 0]), dtype="float32")
identity_image = my_imfilter(test_image, identity_filter)
identity_image2 = vis_hybrid_image(test_image)

cv2.imshow("", identity_image2)
cv2.waitKey(0)
