import numpy as np
import random
from sklearn.neighbors import NearestNeighbors, BallTree


def match_features(features1, features2):
    matches = []
    confidences = []
    ball_tree = BallTree(features1, leaf_size=3)

    dist, ind = ball_tree.query(features2, 2)

    for i in range(len(ind)):
        index = ind[i]
        distances = dist[i]

        if distances[0] / distances[1] < 0.92:
            matches.append([index[0], i])
            confidences.append(1 - distances[0])

    return matches, confidences
