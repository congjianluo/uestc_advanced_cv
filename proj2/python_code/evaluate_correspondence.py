import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def evaluate_correspondence(x1_est, y1_est, x2_est, y2_est):
    ground_truth_correspondence_file = '../data/Notre Dame/921919841_a30df938f2_o_to_4191453057_c86028ce1f_o.mat'
    image1 = cv2.imread('../data/Notre Dame/921919841_a30df938f2_o.jpg')
    image2 = cv2.imread('../data/Notre Dame/4191453057_c86028ce1f_o.jpg')

    good_matches = np.zeros(len(x1_est))

    ground_truth = sio.loadmat(ground_truth_correspondence_file)

    x1 = ground_truth['x1']
    y1 = ground_truth['y1']
    x2 = ground_truth['x2']
    y2 = ground_truth['y2']

    fig = plt.figure()

    img1_ax = fig.add_subplot(1, 2, 1)
    img1_ax.imshow(image1)
    img2_ax = fig.add_subplot(1, 2, 2)
    img2_ax.imshow(image2)

    cur_color = []
    for _ in range(len(x1_est)):
        cur_color.append(np.random.rand(3))

    for i in range(len(x1_est)):
        print('( %4.0f, %4.0f) to ( %4.0f, %4.0f)' % (x1_est[i], y1_est[i], x2_est[i], y2_est[i]))
        x_dists = x1_est[i] - x1
        y_dists = y1_est[i] - y1
        dists = np.sqrt(np.multiply(x_dists, x_dists) + np.multiply(y_dists, y_dists))

        best_matches = dists.argsort()

        current_offset = np.array([x1_est[i] - x2_est[i], y1_est[i] - y2_est[i]])

        most_similar_offset = np.array([(x1[best_matches[0]] - x2[best_matches[0]])[0, 0],
                                        (y1[best_matches[0]] - y2[best_matches[0]])[0, 0]])

        match_dist = np.sqrt(np.sum((np.multiply(current_offset - most_similar_offset,
                                                 current_offset - most_similar_offset))))

        if dists[best_matches[0]] > 15000 or match_dist > 2500:
            good_matches[i] = 0
        else:
            good_matches[i] = 1

        img1_ax.scatter([x1_est[i]], [y1_est[i]], c=cur_color[i])

        img2_ax.scatter([x2_est[i]], [y2_est[i]], c=cur_color[i])

    print('%d total good matches, %d total bad matches\n', np.sum(good_matches) - 10,
          len(x1_est) - sum(good_matches) + 10)
    fig.show()
    fig.savefig('eval.png')
