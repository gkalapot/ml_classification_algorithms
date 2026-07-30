import numpy as np


def compute_euclidean_distance(x1, x2):
    """
    TODO: Compute Euclidean distance between x1 and x2.
    """
    differences = x1 - x2
    sq_differences = differences ** 2
    all_sums = np.sum(sq_differences)
    return np.sqrt(all_sums)


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        """
        TODO: Store the training data.
        """
        self.X_train = X
        self.y_train = y

    def get_k_neighbors_indices(self, x_single):
        """
        TODO:
        Given a single test example x_single, calculate its distance to every point
        in self.X_train and return the INDICES of the k nearest neighbors.
        """
        distances = []

        for i in range(len(self.X_train)):
            distance = compute_euclidean_distance(x_single, self.X_train[i])
            distances.append(distance)

        k_nearest_indices = []

        for _ in range(self.k):
            min_index = 0
            for i in range(1, len(distances)):
                if distances[i] < distances[min_index]:
                    min_index = i

            k_nearest_indices.append(min_index)
            distances[min_index] = float("inf")

        return k_nearest_indices

    def predict(self, X):
        """
        TODO:
        1. Initialize empty predictions.
        2. Loop through every input example in X.
        3. For each example:
           a. Use get_k_neighbors_indices to find the k nearest neighbors.
           b. Get the labels of those neighbors.
           c. Vote (Majority wins).
        """
        preds = []

        for x_single in X:
            neighbor_indices = self.get_k_neighbors_indices(x_single)
            neighbor_labels = []

            for i in neighbor_indices:
                neighbor_labels.append(self.y_train[i])

            positives = 0
            negatives = 0

            for label in neighbor_labels:
                if label == 1.0:
                    positives += 1
                elif label == -1.0:
                    negatives += 1

            if positives >= negatives:
                preds.append(1.0)
            else:
                preds.append(-1.0)

        return preds