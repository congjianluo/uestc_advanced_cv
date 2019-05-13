import cv2
import numpy as np


# This function is intended to behave like the built in function imfilter()
# See 'help imfilter' or 'help conv2'. While terms like "filtering" and
# "convolution" might be used interchangeably, and they are indeed nearly
# the same thing, there is a difference:
# from 'help filter2'
#    2-D correlation is related to 2-D convolution by a 180 degree rotation
#    of the filter matrix.

# Your function should work for color images. Simply filter each color
# channel independently.

# Your function should work for filters of any width and height
# combination, as long as the width and height are odd (e.g. 1, 7, 9). This
# restriction makes it unambigious which pixel in the filter is the center
# pixel.

# Boundary handling can be tricky. The filter can't be centered on pixels
# at the image boundary without parts of the filter being out of bounds. If
# you look at 'help conv2' and 'help imfilter' you see that they have
# several options to deal with boundaries. You should simply recreate the
# default behavior of imfilter -- pad the input image with zeros, and
# return a filtered image which matches the input resolution. A better
# approach is to mirror the image content over the boundaries for padding.

# # Uncomment if you want to simply call imfilter so you can see the desired
# # behavior. When you write your actual solution, you can't use imfilter,
# # filter2, conv2, etc. Simply loop over all the pixels and do the actual
# # computation. It might be slow.
# output = imfilter(image, filter);

def my_imfilter(image, filter):
    ################
    # Your code here
    ################
    filter_width = int(filter.shape[0] / 2)
    filter_height = int(filter.shape[1] / 2)

    image_width = image.shape[0]
    image_height = image.shape[1]

    width_padding = np.zeros([image_width, filter_height], dtype=np.uint8) * 255
    height_padding = np.zeros([filter_width, image_height + filter_height * 2], dtype=np.uint8) * 255

    output = np.ones_like(image)
    print("Filtering...")
    for channel in range(0, 3):
        # 对每个通道进行计算
        channel_data = image[:, :, channel]
        # 上下的padding
        channel_data = np.concatenate([width_padding, channel_data], axis=1)
        channel_data = np.concatenate([channel_data, width_padding], axis=1)
        # 左右的padding
        channel_data = np.concatenate([height_padding, channel_data], axis=0)
        channel_data = np.concatenate([channel_data, height_padding], axis=0)

        # 让filter划过整个通道的长宽
        for column in range(filter_width, channel_data.shape[0] - filter_width):
            for row in range(filter_height, channel_data.shape[1] - filter_height):
                ret = np.multiply(filter, channel_data[column - filter_width:column + filter_width + 1,
                                          row - filter_height:row + filter_height + 1])
                # 保存对应位
                output[column - filter_width, row - filter_height, channel] = min(max(int(np.sum(ret)), 0), 255)

    # 这部分是库函数返回的结果
    print("End...")
    # temp = cv2.filter2D(image, -1, filter)
    # output = temp
    return output
