import numpy as np


def get_bags_of_sifts(image_paths):
    np.load('vocab.mat')
    vocab_size = image_paths.shape[1]
