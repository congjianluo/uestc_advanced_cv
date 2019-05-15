import numpy as np
import os


# Starter code prepared by James Hays for CS 143, Brown University

# This function returns cell arrays containing the file path for each train
# and test image, as well as cell arrays with the label of each train and
# test image. By default all four of these arrays will be 1500x1 where each
# entry is a char array (or string).
def get_image_paths(data_path, categories, num_train_per_cat):
    num_categories = len(categories)  # number of scene categories.
    # This paths for each training and test image. By default it will have 1500
    # entries (15 categories * 100 training and test examples each)
    train_image_paths = [None] * num_categories * num_train_per_cat
    test_image_paths = [None] * num_categories * num_train_per_cat

    # The name of the category for each training and test image. With the
    # default setup, these arrays will actually be the same, but they are built
    # independently for clarity and ease of modification.
    train_labels = [None] * num_categories * num_train_per_cat
    test_labels = [None] * num_categories * num_train_per_cat

    for i in range(0, num_categories):
        images = os.listdir(os.path.join(data_path, "train", categories[i]))

        for j in range(0, num_train_per_cat):
            train_image_paths[i * num_train_per_cat + j] = \
                os.path.join(data_path, 'train', categories[i], images[j])
            train_labels[i * num_train_per_cat + j] = categories[i]

        for j in range(0, num_train_per_cat):
            test_image_paths[i * num_train_per_cat + j] = \
                os.path.join(data_path, 'test', categories[i], images[j])
            test_labels[i * num_train_per_cat + j] = categories[i]

    return train_image_paths, test_image_paths, train_labels, test_labels
