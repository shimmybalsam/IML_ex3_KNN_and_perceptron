import numpy as np
import matplotlib.pyplot as plt
import perceptron
from sklearn.svm import SVC

K = 10000
training_sizes = [5, 10, 15, 25, 70]

def compare_D1():
    mean_perception_accuracy = []
    mean_svm_accuracy = []
    w_D1 = [0.3, -0.5]
    for m in training_sizes:
        perceptron_accuracy = 0
        svm_accuracy = 0
        for i in range(500):
            mone = 0
            training_points = np.random.multivariate_normal(np.zeros(2),
                                                            np.eye(2), size=m)
            training_classified = np.sign(np.dot(training_points, w_D1))
            while 1 not in training_classified or -1 not in training_classified:
                training_points = np.random.multivariate_normal(np.zeros(2),
                                                                np.eye(2),
                                                                size=m)
                training_classified = np.sign(np.dot(training_points, w_D1))

            p = perceptron.Perceptron()
            p.fit(training_points, training_classified)
            testing_points = np.random.multivariate_normal(np.zeros(2),
                                                           np.eye(2), size=K)
            testing_classified = []
            for x in testing_points:
                testing_classified.append(np.sign(np.dot(x,w_D1)))
                if p.predict(x) == np.sign(np.dot(x,w_D1)):
                    mone += 1
            temp = mone/K
            perceptron_accuracy += temp

            svm = SVC(C=1e10, kernel='linear')
            svm.fit(training_points,training_classified)
            svm_accuracy += svm.score(testing_points,testing_classified)

        mean_perception_accuracy.append(perceptron_accuracy/500)
        mean_svm_accuracy.append(svm_accuracy/500)

    plt.plot(training_sizes,mean_perception_accuracy, label="mean_perception")
    plt.plot(training_sizes,mean_svm_accuracy, label="mean_svm")
    plt.title("Distribution D1")
    plt.xlabel("m")
    plt.ylabel("mean accuracy")
    plt.legend(loc=4)
    plt.show()

def get_points_from_D2(size):
    top_left1 = [-3, 3]
    bottom_right1 = [1, 1]
    top_left2 = [-1, -1]
    bottom_right2 = [3, -3]
    x = np.zeros((size,2))
    y = np.zeros(size)
    random_matrix = np.random.rand(size, 3)
    for i in range(len(random_matrix)):
        if random_matrix[i][0] >= 0.5:
            y[i] = 1
            x[i] = [-3 + random_matrix[i][1]*4, 1 + random_matrix[i][2]*2]
        else:
            y[i] = -1
            x[i] = [-1 + random_matrix[i][1]*4, -3 + random_matrix[i][2]*2]
    return x, y

def compare_D2():
    mean_perception_accuracy = []
    mean_svm_accuracy = []

    for m in training_sizes:
        perceptron_accuracy = 0
        svm_accuracy = 0
        for i in range(500):
            mone = 0
            training_points, training_classified = get_points_from_D2(m)
            while 1 not in training_classified or -1 not in training_classified:
                training_points, training_classified = get_points_from_D2(m)
            p = perceptron.Perceptron()
            p.fit(training_points, training_classified)

            testing_points, testing_classified = get_points_from_D2(K)
            for i in range(len(testing_points)):
                if p.predict(testing_points[i]) == testing_classified[i]:
                    mone += 1
            temp = mone/K
            perceptron_accuracy += temp

            svm = SVC(C=1e10, kernel='linear')
            svm.fit(training_points,training_classified)
            svm_accuracy += svm.score(testing_points,testing_classified)
        mean_perception_accuracy.append(perceptron_accuracy / 500)
        mean_svm_accuracy.append(svm_accuracy / 500)

    plt.plot(training_sizes, mean_perception_accuracy, label="mean_perception")
    plt.plot(training_sizes, mean_svm_accuracy, label="mean_svm")
    plt.title("Distribution D2")
    plt.xlabel("m")
    plt.ylabel("mean accuracy")
    plt.legend(loc=4)
    plt.show()

# compare_D1()
compare_D2()

