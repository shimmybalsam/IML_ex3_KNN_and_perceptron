import numpy as np
from sklearn.svm import SVC
import perceptron
import matplotlib.pyplot as plt
import random

SAMPELS = [5, 10, 15, 25, 70]
K = 10000
REPEATS = 50

DIM = 2
rec_1_left = [-3, 3]
rec_1_right = [1, 1]
rec_2_left = [-1, -1]
rec_2_right = [3, -3]
X = 0
Y = 1

mean = [0, 0]
cov = [[1,0],[0,1]]
w = (0.3, -0.5)


def distribution_1(num_of_samples):
    training_point = np.random.multivariate_normal(mean, cov, num_of_samples)
    true_labels_training = np.sign(np.dot(training_point, w))
    return training_point, true_labels_training


def distribution_2(num_of_samples):
    x = np.zeros((num_of_samples, DIM))
    y = np.zeros(num_of_samples)
    for i in range(num_of_samples):
        if random.randint(0, 1):
            x[i] = [random.uniform(rec_1_left[X], rec_1_right[X]),
                          random.uniform(rec_1_right[Y], rec_1_left[Y])]
            y[i] = 1
        else:
            x[i] = [random.uniform(rec_2_left[X], rec_2_right[X]),
                          random.uniform(rec_2_right[Y], rec_2_left[Y])]
            y[i] = -1
    return x, y


def main(distribution_name, distribution_function):
    svm_mean_accuracy = []
    perceptron_mean_accuracy = []
    for m in SAMPELS:
        perceptron_accuracy = 0
        svm_accuracy = 0

        for i in range(REPEATS):
            training_point, true_labels_training = distribution_function(m)
            while len(plt.np.unique(true_labels_training)) <= 1:
                training_point, true_labels_training = distribution_function(m)

            test_point, true_labels_test = distribution_function(K)
            while len(plt.np.unique(true_labels_test)) <= 1:
                test_point, true_labels_test = distribution_function(K)

            my_perc = perceptron.Perceptron()
            my_perc.fit(training_point, true_labels_training)

            perceptron_accuracy = perceptron_accuracy + accuracy_calc(my_perc, test_point, true_labels_test)

            svm1 = SVC(C= 1e10,kernel = 'linear')
            svm1.fit(training_point, true_labels_training)
            svm_accuracy = svm_accuracy + svm1.score(test_point, true_labels_test)
            svm1.score(test_point, true_labels_test)

        svm_mean_accuracy.append(svm_accuracy / REPEATS)
        perceptron_mean_accuracy.append(perceptron_accuracy / REPEATS)

    print(svm_mean_accuracy)
    print(perceptron_mean_accuracy)
    plt.plot(SAMPELS, svm_mean_accuracy, label="svm mean accuracy")
    plt.plot(SAMPELS, perceptron_mean_accuracy,label="perceptron mean accuracy")
    plt.title(distribution_name)
    plt.legend()
    plt.savefig(distribution_name)
    plt.show()
    plt.clf()

def accuracy_calc(perc, x, y):
    acc = 0
    n = len(y)
    y_np = np.asarray(y)
    for i in range(n):
        if int(perc.predict(x[i])) == int(y_np[i]):
            acc += 1
    mean_acc = acc / n
    return mean_acc

if __name__ == '__main__':
    main("distribution 1", distribution_1)
    main("distribution 2", distribution_2)