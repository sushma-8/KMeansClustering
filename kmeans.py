import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib import style
import random as rd

style.use('ggplot')

K = 3


class KMeansClustering:
    def __init__(self, data):
        self.data = data
        # Number of training data
        self.points_count = data.shape[0]
        # Number of features in the data
        self.axis = data.shape[1]
        self.k = K

    def kmeans_algorithm(self):

        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0)
        centers = np.random.randn(self.k, self.axis) * std + mean

        centers_old = np.zeros(centers.shape)  # to store old centers
        centers_new = deepcopy(centers)  # Store new centers

        clusters = np.zeros(self.points_count)
        distances = np.zeros((self.points_count, self.k))

        error = np.linalg.norm(centers_new - centers_old)

        while error != 0:

            # Measure the distance to every centroid
            for i in range(self.k):
                distances[:, i] = np.linalg.norm(self.data - centers_new[i], axis=1)
            # print(distances)

            # Assign point to closest centroid
            clusters = np.argmin(distances, axis=1)
            centers_old = deepcopy(centers_new)
            # Calculate mean for every cluster and update the center
            for i in range(self.k):
                centers_new[i] = np.mean([self.data[x] for x in range(len(clusters)) if clusters[x] == i], axis=0)

            error = np.linalg.norm(centers_new - centers_old)
            print(error)

        print(centers_new)
        colors = ['orange', 'blue', 'green']
        for i in range(self.points_count):
            plt.scatter(self.data[i, 0], self.data[i, 1], s=40, color=colors[int(clusters[i])])
        plt.scatter(centers_new[:, 0], centers_new[:, 1], marker='*', c='g', s=150)
        plt.show()


if __name__ == "__main__":

    fp = open('clusters.txt', 'r')
    try:
        coordinates = np.loadtxt(fp, delimiter=',')
    except ValueError:
        print('Debug')

    k_means = KMeansClustering(coordinates)
    k_means.kmeans_algorithm()
