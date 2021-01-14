import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


TEST_SIZE = 1000
data = np.loadtxt("spam.data")
np.random.shuffle(data)
testing = data[:TEST_SIZE,:]
training = data[TEST_SIZE:,:]

training_y = training[:,-1]
training_X = training[:,:-1]

testing_y = testing[:,-1]
testing_X = testing[:,:-1]

my_lr = LogisticRegression()
my_lr.fit(training_X,training_y)
predict = my_lr.predict_proba(testing_X)
sort_pred = np.argsort(predict[:,1])
NP = int(sum(testing_y))
NN = TEST_SIZE - NP
Ni_arr = np.zeros(NP)
fpr_arr = np.zeros(NP)

for i in range(1,NP):
    mone = 0
    real_tags_needed = 0
    for j in sort_pred:
        real_tags_needed += 1
        if testing_y[j] == 1:
            mone += 1
            if mone == i:
                break
    Ni_arr[i] = real_tags_needed

for i in range(1,NP):
    fpr_arr[i] = (Ni_arr[i]-i)/NN

x_axis = []
for i in range(NP):
    x_axis.append(i/NP)

plt.plot(x_axis,fpr_arr)
plt.title("ROC curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()
