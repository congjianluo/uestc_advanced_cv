import cv2
import numpy as np


def vis_hybrid_image(hybrid_image):
    scales = 5
    scale_factor = 0.5
    padding = 5

    original_height = hybrid_image.shape[0]
    num_colors = hybrid_image.shape[2]
    output = hybrid_image
    cur_image = hybrid_image

    for i in range(2, scales):
        # temp2 = np.ones((original_height, padding, num_colors), float)
        output = np.concatenate([output, np.ones((original_height, padding, num_colors))], axis=1)
        cur_image = cv2.resize(cur_image, (0, 0), fx=scale_factor, fy=scale_factor)
        tmp = np.concatenate([np.ones((original_height - cur_image.shape[0],
                                       cur_image.shape[1], num_colors)), cur_image], axis=0)
        output = np.concatenate([output, tmp], axis=1)
    return output
