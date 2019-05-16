# Sliding window face detection with linear SVM.
# All code by James Hays, except for pieces of evaluation code from Pascal
# VOC toolkit. Images from CMU+MIT face database, CalTech Web Face
# Database, and SUN scene database.

# Code structure:
# proj4.m <--- You code parts of this
#  + get_positive_features.m  <--- You code this
#  + get_random_negative_features.m  <--- You code this
#   [classifier training]   <--- You code this
#  + report_accuracy.m
#  + run_detector.m  <--- You code this
#    + non_max_supr_bbox.m
#  + evaluate_all_detections.m
#    + VOCap.m
#  + visualize_detections_by_image.m
#  + visualize_detections_by_image_no_gt.m
#  + visualize_detections_by_confidence.m

# Other functions. You don't need to use any of these unless you're trying
# to modify or build a test set:

# Training and Testing data related functions:
# test_scenes/visualize_cmumit_database_landmarks.m
# test_scenes/visualize_cmumit_database_bboxes.m
# test_scenes/cmumit_database_points_to_bboxes.m #This function converts
# from the original MIT+CMU test set landmark points to Pascal VOC
# annotation format (bounding boxes).

# caltech_faces/caltech_database_points_to_crops.m #This function extracts
# training crops from the Caltech Web Face Database. The crops are
# intentionally large to contain most of the head, not just the face. The
# test_scene annotations are likewise scaled to contain most of the head.

# set up paths to VLFeat functions.
# See http://www.vlfeat.org/matlab/matlab.html for VLFeat Matlab documentation
# This should work on 32 and 64 bit versions of Windows, MacOS, and Linux

import os

from get_positive_features import get_positive_features
from get_random_negative_features import get_random_negative_features

try:
    os.mkdir('visualizations')
except Exception as e:
    pass
# change if you want to work with a network copy
data_path = '../data/'
# Positive training examples. 36x36 head crops
train_path_pos = os.path.join(data_path, 'caltech_faces/Caltech_CropFaces')
# We can mine random or hard negatives from here
non_face_scn_path = os.path.join(data_path, 'train_non_face_scenes')
# CMU+MIT test scenes
test_scn_path = os.path.join(data_path, 'test_scenes/test_jpg')
# the ground truth face locations in the test set
label_path = os.path.join(data_path, 'test_scenes/ground_truth_bboxes.txt')

# The faces are 36x36 pixels, which works fine as a template size. You could
# add other fields to this struct if you want to modify HoG default
# parameters such as the number of orientations, but that does not help
# performance in our limited test.
feature_params = {
    'template_size': 36,
    'hog_cell_size': 6
}

## Step 1. Load positive training crops and random negative examples
# YOU CODE 'get_positive_features' and 'get_random_negative_features'

features_pos = get_positive_features(train_path_pos, feature_params)
num_negative_examples = 10000  # Higher will work strictly better, but you should start with 10000 for debugging

features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_negative_examples)

## step 2. Train Classifier
# Use vl_svmtrain on your training features to get a linear classifier
# specified by 'w' and 'b'
# [w b] = vl_svmtrain(X, Y, lambda)
# http://www.vlfeat.org/sandbox/matlab/vl_svmtrain.html
# 'lambda' is an important parameter, try many values. Small values seem to
# work best e.g. 0.0001, but you can try other values

# YOU CODE classifier training. Make sure the outputs are 'w' and 'b'.
