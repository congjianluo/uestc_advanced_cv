# This function will sample SIFT descriptors from the training images,
# cluster them with kmeans, and then return the cluster centers.

import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import random


def build_vocabulary(image_paths, vocab_size):
    cluster_SIFT_features = []
    sift = cv2.xfeatures2d.SIFT_create()
    for image_path in tqdm(image_paths, desc="Imaging-SIFT"):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        locations, SIFT_features = sift.detectAndCompute(gray, None)
        temp = SIFT_features.tolist()
        cluster_SIFT_features += temp
    # cluster_SIFT_features = np.array(cluster_SIFT_features)
    # kmeans = KMeans(n_clusters=vocab_size, random_state=0).fit(cluster_SIFT_features)
    cluster_SIFT_features = random.sample(cluster_SIFT_features, 400)
    kmeans = KMeans(n_clusters=vocab_size, max_iter=100).fit(cluster_SIFT_features)
    cluster_centers = kmeans.cluster_centers_
    return cluster_centers


# The inputs are images, a N x 1 cell array of image paths and the size of
# the vocabulary.

# The output 'vocab' should be vocab_size x 128. Each row is a cluster
# centroid / visual word.

"""
Useful functions:
[locations, SIFT_features] = vl_dsift(img) 
 http://www.vlfeat.org/matlab/vl_dsift.html
 locations is a 2 x n list list of locations, which can be thrown away here
  (but possibly used for extra credit in get_bags_of_sifts if you're making
  a "spatial pyramid").
 SIFT_features is a 128 x N matrix of SIFT features
  note: there are step, bin size, and smoothing parameters you can
  manipulate for vl_dsift(). We recommend debugging with the 'fast'
  parameter. This approximate version of SIFT is about 20 times faster to
  compute. Also, be sure not to use the default value of step size. It will
  be very slow and you'll see relatively little performance gain from
  extremely dense sampling. You are welcome to use your own SIFT feature
  code! It will probably be slower, though.

[centers, assignments] = vl_kmeans(X, K)
 http://www.vlfeat.org/matlab/vl_kmeans.html
  X is a d x M matrix of sampled SIFT features, where M is the number of
   features sampled. M should be pretty large! Make sure matrix is of type
   single to be safe. E.g. single(matrix).
  K is the number of clusters desired (vocab_size)
  centers is a d x K matrix of cluster centroids. This is your vocabulary.
   You can disregard 'assignments'.

  Matlab has a build in kmeans function, see 'help kmeans', but it is
  slower.
"""

# Load images from the training set. To save computation time, you don't
# necessarily need to sample from all images, although it would be better
# to do so. You can randomly sample the descriptors from each image to save
# memory and speed up the clustering. Or you can simply call vl_dsift with
# a large step size here, but a smaller step size in make_hist.m.

# For each loaded image, get some SIFT features. You don't have to get as
# many SIFT features as you will in get_bags_of_sift.m, because you're only
# trying to get a representative sample here.

# Once you have tens of thousands of SIFT features from many training
# images, cluster them with kmeans. The resulting centroids are now your
# visual word vocabulary.
