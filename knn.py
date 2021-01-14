import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt


class Knn:

    def __init__(self,k):
        self.k = k
        self.X = []
        self.y = []

    def fit(self,X,y):
        self.X = X
        self.y = y

    def predict(self,x):
        dist_sort = np.argsort(euclidean_distances([x], self.X))[0]
        k_nearest = dist_sort[:self.k]
        k_nearest_labels = []
        for i in k_nearest:
            k_nearest_labels.append(self.y[i])
        if (sum(k_nearest_labels)/self.k) > 0.5:
            return 1
        else:
            return 0


def compare():
    TEST_SIZE = 1000
    num_of_neighbors = [1,2,5,10,100]
    loss = [0] * (len(num_of_neighbors))

    data = np.loadtxt("spam.data")
    np.random.shuffle(data)
    testing = data[:TEST_SIZE, :]
    training = data[TEST_SIZE:, :]

    training_y = training[:, -1]
    training_X = training[:, :-1]

    testing_y = testing[:, -1]
    testing_X = testing[:, :-1]

    for k in range(len(num_of_neighbors)):
        my_knn = Knn(num_of_neighbors[k])
        my_knn.fit(training_X,training_y)
        mone = 0
        for j in range(TEST_SIZE):
            if my_knn.predict(testing_X[j]) != testing_y[j]:
                mone += 1
        loss[k] += mone/TEST_SIZE

    plt.plot(num_of_neighbors,loss)
    plt.title("Test error per knn")
    plt.xlabel("Num of neighbors")
    plt.ylabel("Test error")
    plt.show()

compare()







