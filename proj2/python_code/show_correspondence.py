import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


def show_image_point(image, x1, y1):
    plt.imshow(image)
    plt.scatter(x1, y1)
    plt.show()


def show_correspondence(image1, image2, X1, Y1, X2, Y2):
    colors = np.random.rand(len(X1))
    plt.imshow(image1)

    print('Saving visualization to vis.jpg\n')
    plt.scatter(X1, Y1, s=10, c=colors)
    plt.show()

    plt.imshow(image2)
    plt.scatter(X2, Y2, s=10, c=colors)
    plt.show()
