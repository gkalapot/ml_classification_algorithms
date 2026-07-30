import numpy as np


class MostFrequentClassClassifier:
    def __init__(self):
        self.prediction = 0

    def train(self, X, y):
        """
        TODO: Find the most frequent label in y and store it in self.prediction.
        """
        positives = 0
        negatives = 0

        for i in y:
            if i == 1.0:
                positives += 1
            elif i == -1.0:
                negatives += 1

        if positives >= negatives:
            self.prediction = 1.0
        else:
            self.prediction = -1.0

    def predict(self, X):
        """
        TODO: Return a vector of predictions, all equal to self.prediction.
        """
        predictions = []

        for i in range(X.shape[0]):
            predictions.append(self.prediction)

        return np.array(predictions)
