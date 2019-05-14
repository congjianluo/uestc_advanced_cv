import numpy as np
import os


def get_image_paths(data_path, categories, num_train_per_cat):
    num_categories = len(categories)
    train_image_paths = np.ones([num_categories * num_train_per_cat])
    test_image_paths = np.ones([num_categories * num_train_per_cat])

    train_labels = np.ones([num_categories * num_train_per_cat])
    test_labels = np.ones([num_categories * num_train_per_cat])

    for i in range(0, num_categories):
        images = os.listdir(os.path.join(data_path, "train", categories[i], ".jpg"))

        for j in range(0, num_train_per_cat):
            train_image_paths[i * num_train_per_cat + j] = ""
            train_labels[i * num_train_per_cat + j] = categories[i]

        for j in range(0, num_train_per_cat):
            test_image_paths[i * num_train_per_cat + j] = ""
            test_labels[i * num_train_per_cat + j] = categories[i]
