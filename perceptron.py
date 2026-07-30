import numpy as np


class Perceptron:
    def __init__(self, num_epochs=10):
        self.num_epochs = num_epochs
        self.w = None
        self.b = 0

    def train(self, X, y):
        """
        TODO: Implement the Perceptron Update Rule.
        1. Init w and b to zeros (w is a vector and b is a scalar).
        2. Loop epochs.
        3. Loop examples:
           If prediction is wrong:
              w = w + y * x
              b = b + y
        """
        self.w = []

        for i in range(X.shape[1]):
            self.w.append(0.0)

        self.b = 0.0

        for k in range(self.num_epochs):
            for i in range(len(X)):
                x_i = X[i]
                y_i = y[i]

                score = np.dot(self.w, x_i) + self.b
                if y_i * score <= 0:
                    self.w = self.w + y_i * x_i
                    self.b = self.b + y_i

    def predict(self, X):
        """
        TODO: Compute w*x + b. Return +1 or -1.
        """
        predictions = []
        for x_single in X:
            score = np.dot(self.w, x_single) + self.b
            if score >= 0:
                predictions.append(1.0)
            else:
                predictions.append(-1.0)

        return np.array(predictions)
