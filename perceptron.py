import numpy as np


class Perceptron:

    def __init__(self):
        self.w = np.zeros(1)

    def fit(self, X, y):
        d = len(X[0])
        self.w = np.zeros(d)
        my_x = np.asarray(X)
        my_y = np.asarray(y)
        t = True
        while t:
            t = False
            checker = np.dot(my_x, self.w)
            for i in range(len(X)):
                if np.dot(checker[i], my_y[i]) <= 0:
                    self.w = np.add(self.w, np.dot(my_x[i], my_y[i]))
                    t = True
                    break

    def predict(self, x):
        result = np.dot(self.w, x)
        if result >= 0:
            return 1
        else:
            return -1




