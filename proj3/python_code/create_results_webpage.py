# Starter code prepared by James Hays for CS 143, Brown University

# This function creates a webpage (html and images) visualizing the
# classiffication results. This webpage will contain
# (1) A confusion matrix plot
# (2) A table with one row per category, with 3 columns - training
# examples, true positives, false positives, and false negatives.

# false positives are instances claimed as that category but belonging to
# another category, e.g. in the 'forest' row an image that was classified
# as 'forest' but is actually 'mountain'. This same image would be
# considered a false negative in the 'mountain' row, because it should have
# been claimed by the 'mountain' classifier but was not.

# This webpage is similar to the one we created for the SUN database in
# 2010: http://people.csail.mit.edu/jxiao/SUN/classification397.html
import os
import numpy as np


def create_results_webpage(train_image_paths, test_image_paths,
                           train_labels, test_labels, categories, abbr_categories, predicted_categories):
    print("Creating results_webpage/index.html, thumbnails, and confusion matrix")
    num_samples = 2
    thumbnail_height = 75
    os.remove('results_webpage/thumbnails/*.jpg')
    os.mkdir('results_webpage')
    os.mkdir('results_webpage/thumbnails')

    with open('results_webpage/index.html', 'w+t') as f:
        num_categories = len(categories)
        confusion_matrix = np.zeros([num_categories, num_categories])

        for i in range(predicted_categories):
            row = test_labels[i]
            column = test_labels[i]
            confusion_matrix[row, column] = confusion_matrix[row, column] + 1

        num_test_per_cat = len(test_labels) / num_categories
        confusion_matrix = np.divide(confusion_matrix, num_test_per_cat)
        accuracy = np.mean(confusion_matrix)
    print('Accuracy (mean of diagonal of confusion matrix) is %.3f\n' % accuracy)


