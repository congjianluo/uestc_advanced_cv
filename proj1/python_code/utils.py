import cv2
import os


def luo_imshow(name, image):
    cv2.imshow(name, image)
    # cv2.waitKey(0)


def luo_imwrite(image, name):
    path = os.path.join("../experience_results", name)
    cv2.imwrite(path, image)
