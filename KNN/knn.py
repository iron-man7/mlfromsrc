
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.X = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def euclidean_dist(self, x, y):
        dist = np.sqrt(np.sum(x -y) ** 2)
        return dist

    def predict(self, X):
        pred = [self.neighbours(x) for x in X]
        return pred

    def neighbours(self, x):

        # calculate euclidean distance in space
        distances = [ self.euclidean_dist(x, y) for y in self.X ]
        #print("distances", distances)

        # argsort
        k_indices = np.argsort(distances)[:self.k]
        #print("k_indices", k_indices)
        k_near_labels = [self.y[i] for i in k_indices]
        #print("labels", k_near_labels)

        most_common = Counter(k_near_labels).most_common()
        #print("most_common", most_common)
        return most_common[0][0]



