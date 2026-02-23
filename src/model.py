import numpy as np

class LinearRegressionScratch:
    def __init__(self, lr=0.01, iterations=1500):
        self.lr = lr
        self.iterations = iterations

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.W = np.zeros((self.n, 1))
        self.b = 0
        self.cost_history = []

        for _ in range(self.iterations):
            y_pred = np.dot(X, self.W) + self.b

            dW = (1 / self.m) * np.dot(X.T, (y_pred - y))
            db = (1 / self.m) * np.sum(y_pred - y)

            self.W -= self.lr * dW
            self.b -= self.lr * db

            cost = (1 / (2 * self.m)) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)

    def predict(self, X):
        return np.dot(X, self.W) + self.b