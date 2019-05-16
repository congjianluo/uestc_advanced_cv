import numpy as np
import random


def match_features(features1, features2):
    num_features1 = features1.shape[0]
    num_features2 = features2.shape[0]

    matches = np.zeros(num_features1, 2)
    random.shuffle(num_features1)
    random.shuffle(num_features2)

    matches[:, 0] = num_features1
    matches[:, 1] = num_features2

    # Placeholder that you can delete. Random matches and confidences
    confidences = np.random.rand(num_features1, 1)

    # Sort the matches so that the most confident onces are at the top of the
    # list. You should probably not delete this, so that the evaluation
    # functions can be run on the top matches easily.

    np.sort(confidences)

    return matches, confidences
