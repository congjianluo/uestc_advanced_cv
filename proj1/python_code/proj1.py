import cv2

image1 = cv2.imread('../data/dog.bmp')
image2 = cv2.imread('../data/cat.bmp')

cutoff_frequency = 7

filter = cv2.fspecial('Gaussian', cutoff_frequency*4+1, cutoff_frequency)
# imshow(low_frequencies)
