import cv2
import numpy as np


# visualize a hybrid image by progressively downsampling the image and
# concatenating all of the images together.
def vis_hybrid_image(hybrid_image):
    scales = 5  # how many downsampled versions to create
    scale_factor = 0.5  # how much to downsample each time
    padding = 5  # how many pixels to pad.

    original_height = hybrid_image.shape[0]
    num_colors = hybrid_image.shape[2]  # counting how many color channels the input has
    output = hybrid_image
    cur_image = hybrid_image

    for i in range(2, scales):
        # add padding
        output = np.concatenate([
            output, np.zeros([original_height, padding, num_colors], dtype=np.uint8)
        ], axis=1)
        # dowsample image;
        cur_image = cv2.resize(cur_image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        # pad the top and append to the output
        output = np.concatenate(
            [
                output,
                np.concatenate(
                    [
                        np.zeros(
                            [original_height - cur_image.shape[0],
                             cur_image.shape[1], num_colors], dtype=np.uint8
                        ),
                        cur_image
                    ],
                    axis=0
                )
            ], axis=1)
    return output

# code by CongjianLuo
