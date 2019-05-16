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
import matplotlib.pyplot as plt
import random
import cv2
import shutil


def my_strcmp(string, str_list):
    result = []

    for item in str_list:
        if item == string:
            result.append(1)
        else:
            result.append(0)
    return result


def create_results_webpage(train_image_paths, test_image_paths,
                           train_labels, test_labels, categories, abbr_categories, predicted_categories):
    print("Creating results_webpage/index.html, thumbnails, and confusion matrix")
    num_samples = 2
    thumbnail_height = 75
    try:
        shutil.rmtree('results_webpage')
    except Exception as e:
        pass

    try:
        os.mkdir('results_webpage')
        os.mkdir('results_webpage/thumbnails')
    except Exception as e:
        pass

    with open('results_webpage/index.html', 'w+t') as f:
        num_categories = len(categories)
        confusion_matrix = np.zeros([num_categories, num_categories])

        for i in range(len(predicted_categories)):
            row = categories.index(test_labels[i])
            column = categories.index(predicted_categories[i])
            confusion_matrix[row, column] = confusion_matrix[row, column] + 1

        num_test_per_cat = len(test_labels) / num_categories
        confusion_matrix = np.divide(confusion_matrix, num_test_per_cat)
        accuracy = np.mean(np.diag(confusion_matrix))
        print('Accuracy (mean of diagonal of confusion matrix) is %.3f\n' % accuracy)

        plt.matshow(confusion_matrix)
        plt.savefig("results_webpage/confusion_matrix.png")
        plt.show()
        f.write('<!DOCTYPE html>\n')
        f.write('<!DOCTYPE html>\n')
        f.write('<html>\n')
        f.write('<head>\n')
        f.write(
            '<link href=''http://fonts.googleapis.com/css?family=Nunito:300|Crimson+Text|Droid+Sans+Mono'' rel=''stylesheet'' type=''text/css''>\n')
        f.write('<style type="text/css">\n')

        f.write('body {\n')
        f.write('  margin: 0px\n')
        f.write('  width: 100%%\n')
        f.write('  font-family: ''Crimson Text'', serif\n')
        f.write('  background: #fcfcfc\n')
        f.write('}\n')
        f.write('table td {\n')
        f.write('  text-align: center\n')
        f.write('  vertical-align: middle\n')
        f.write('}\n')
        f.write('h1 {\n')
        f.write('  font-family: ''Nunito'', sans-serif\n')
        f.write('  font-weight: normal\n')
        f.write('  font-size: 28px\n')
        f.write('  margin: 25px 0px 0px 0px\n')
        f.write('  text-transform: lowercase\n')
        f.write('}\n')
        f.write('.container {\n')
        f.write('  margin: 0px auto 0px auto\n')
        f.write('  width: 1160px\n')
        f.write('}\n')

        f.write('</style>\n')
        f.write('</head>\n')
        f.write('<body>\n\n')

        f.write('<div class="container">\n\n\n')
        f.write('<center>\n')
        f.write('<h1>CS 143 Project 3 results visualization</h1>\n')
        f.write('<img src="confusion_matrix.png">\n\n')
        f.write('<br>\n')
        f.write('Accuracy (mean of diagonal of confusion matrix) is %.3f\n' % accuracy)
        f.write('<p>\n\n')

        # Create results table
        f.write('<table border=0 cellpadding=4 cellspacing=1>\n')
        f.write('<tr>\n')
        f.write('<th>Category name</th>\n')
        f.write('<th>Accuracy</th>\n')
        f.write('<th colspan=%d>Sample training images</th>\n' % num_samples)
        f.write('<th colspan=%d>Sample true positives</th>\n' % num_samples)
        f.write('<th colspan=%d>False positives with true label</th>\n' % num_samples)
        f.write('<th colspan=%d>False negatives with wrong predicted label</th>\n' % num_samples)
        f.write('</tr>\n')

        for i in range(num_categories):
            f.write('<tr>\n')

            f.write('<td>')  # category name
            f.write('{}'.format(categories[i]))
            f.write('</td>\n')

            f.write('<td>')  # category accuracy
            f.write('%.3f' % confusion_matrix[i, i])
            f.write('</td>\n')

            train_examples_index = my_strcmp(categories[i], train_labels)
            train_examples = [train_image_paths[index] for index in
                              range(len(train_examples_index)) if train_examples_index[index] == 1]

            true_positives_index = np.logical_and(
                my_strcmp(categories[i], test_labels)
                , my_strcmp(categories[i], predicted_categories)
            )
            true_positives = [test_image_paths[index] for index in
                              range(len(true_positives_index)) if index == 1]

            false_positive_inds = np.logical_and(
                np.logical_not(my_strcmp(categories[i], test_labels))
                , my_strcmp(categories[i], predicted_categories))
            false_positives = [test_image_paths[index] for index in
                               range(len(false_positive_inds)) if index == 1]
            false_positive_labels = [test_labels[index] for index in
                                     range(len(false_positive_inds)) if index == 1]

            false_negative_inds = np.logical_and(
                my_strcmp(categories[i], test_labels),
                np.logical_not(my_strcmp(categories[i], predicted_categories))
            )
            false_negatives = [test_image_paths[index] for index in
                               range(len(false_negative_inds)) if index == 1]
            false_negative_labels = [predicted_categories[index] for index in
                                     range(len(false_positive_inds)) if index == 1]

            random.shuffle(train_examples)
            random.shuffle(true_positives)

            random.shuffle(false_positives)
            random.shuffle(false_positive_labels)

            random.shuffle(false_negatives)
            random.shuffle(false_negative_labels)

            train_examples = train_examples[:min(len(train_examples), num_samples)]
            true_positives = true_positives[:min(len(true_positives), num_samples)]
            false_positives = false_positives[:min(len(false_positives), num_samples)]
            false_positive_labels = false_positive_labels[:min(len(false_positive_labels), num_samples)]
            false_negatives = false_negatives[:min(len(false_negatives), num_samples)]
            false_negative_labels = false_negative_labels[:min(len(false_negative_labels), num_samples)]

            for j in range(num_samples):
                if j < len(train_examples):
                    tmp = cv2.imread(train_examples[j])
                    height = tmp.shape[0]
                    rescale_factor = thumbnail_height / height
                    tmp = cv2.resize(tmp, None, fx=rescale_factor, fy=rescale_factor)
                    height, width = tmp.shape[:2]

                    dir, name = os.path.split(train_examples[j])

                    cv2.imwrite(os.path.join('results_webpage/thumbnails/', categories[i] + '_' + name),
                                tmp, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                    f.write('<td bgcolor=LightBlue>')
                    f.write('<img src="{}" width=%d height=%d>'.format(
                        os.path.join('thumbnails/', categories[i] + '_' + name), width,
                        height))
                    f.write('</td>\n')
                else:
                    f.write('<td bgcolor=LightBlue>')
                    f.write('</td>\n')

            for j in range(num_samples):
                if j < len(true_positives):
                    tmp = cv2.imread(true_positives[j])
                    height = tmp.shape[0]
                    rescale_factor = thumbnail_height / height
                    tmp = cv2.resize(tmp, None, fx=rescale_factor, fy=rescale_factor)
                    height, width = tmp.shape[:2]

                    dir, name = os.path.split(true_positives[j])

                    cv2.imwrite(os.path.join('results_webpage/thumbnails/', categories[i] + '_' + name),
                                tmp, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                    f.write('<td bgcolor=LightGreen>')
                    f.write('<img src="{}" width=%d height=%d>'.format(
                        os.path.join('thumbnails/', categories[i] + '_' + name), width,
                        height))
                    f.write('</td>\n')
                else:
                    f.write('<td bgcolor=LightGreen>')
                    f.write('</td>\n')

            for j in range(num_samples):
                if j < len(true_positives):
                    tmp = cv2.imread(true_positives[j])
                    height = tmp.shape[0]
                    rescale_factor = thumbnail_height / height
                    tmp = cv2.resize(tmp, None, fx=rescale_factor, fy=rescale_factor)
                    height, width = tmp.shape[:2]

                    dir, name = os.path.split(true_positives[j])

                    cv2.imwrite(os.path.join('results_webpage/thumbnails/', categories[i] + '_' + name),
                                tmp, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                    f.write('<td bgcolor=LightGreen>')
                    f.write('<img src="{}" width=%d height=%d>'.format(
                        os.path.join('thumbnails/', categories[i] + '_' + name), width,
                        height))
                    f.write('</td>\n')
                else:
                    f.write('<td bgcolor=LightGreen>')
                    f.write('</td>\n')

            for j in range(num_samples):
                if j < len(false_positives):
                    tmp = cv2.imread(false_positives[j])
                    height, width = tmp.shape[:2]
                    rescale_factor = thumbnail_height / height
                    tmp = cv2.resize(tmp, None, fx=rescale_factor, fy=rescale_factor)
                    height, width = tmp.shape[:2]

                    dir, name = os.path.split(false_positives[j])

                    cv2.imwrite(os.path.join('results_webpage/thumbnails/', categories[i] + '_' + name),
                                tmp, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                    f.write('<td bgcolor=LightCoral>')
                    f.write('<img src="{}" width=%d height=%d>'.format(
                        os.path.join('thumbnails/', categories[i] + '_' + name), width,
                        height))
                    f.write('</td>\n')
                else:
                    f.write('<td bgcolor=LightCoral>')
                    f.write('</td>\n')

        f.write("</tr>\n")
        f.write('<tr>\n')
        f.write('<th>Category name</th>\n')
        f.write('<th>Accuracy</th>\n')
        f.write('<th colspan=%d>Sample training images</th>\n' % num_samples)
        f.write('<th colspan=%d>Sample true positives</th>\n' % num_samples)
        f.write('<th colspan=%d>False positives with true label</th>\n' % num_samples)
        f.write('<th colspan=%d>False negatives with wrong predicted label</th>\n' % num_samples)
        f.write('</tr>\n')

        f.write('</table>\n')
        f.write('</center>\n\n\n')
        f.write('</div>\n')

        f.write('</body>\n')
        f.write('</html>\n')
