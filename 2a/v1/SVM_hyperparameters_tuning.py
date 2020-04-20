"""
In this file, we try to optimize the parameter of a support vector machine using a gaussian kernel.
Since the SVM_linearVSgaussian.py file, we have observe that the gaussian kernel may provide the best result.
"""

import numpy as np
from sklearn import svm
import csv
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score

# First, we build our training set
with open('mnist_train.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    matrix = np.array(data, dtype=int)
    training_set = matrix[:, 1:]
    labels_training = matrix[:, 0]


# For the model selection part, we will not work on all the data set, this could take too much time.
training_length = 20000
test_length = 3333
training_set = training_set[:training_length, :]
labels_training = labels_training[:training_length]


# First we normalize our data :
training_set = training_set/255.0


# And we scale it :
training_set_scaled = scale(training_set)


# Now, since our data set is ready, we can find the best parameters for our model
# For the gamma value, we will test 10^-3, 10^-2, 10^-1, 1 and the standard parameters
# For the C value, we will test 1, 10, 100, 1000

k = 0
l = -3
while k < 4:
    c = 10**k
    model_gaussian = svm.SVC(C=c, kernel='rbf', gamma='auto')
    scores_gaussian = cross_val_score(model_gaussian, training_set_scaled, labels_training, cv=5)
    print("The cross validation score with a C value of {} and a  standard gamma value is :".format(c))
    print(scores_gaussian)
    print("Which mean is :")
    print(np.mean(scores_gaussian))
    l = -3
    while l < 1:
        gam = 10**l
        model_gaussian = svm.SVC(C=c, kernel='rbf', gamma=gam)
        scores_gaussian = cross_val_score(model_gaussian, training_set_scaled, labels_training, cv=5)
        print("The cross validation score with a C value of {} and a gamma value of {} is :".format(c, gam))
        print(scores_gaussian)
        print("Which mean is :")
        print(np.mean(scores_gaussian))
        l = l+1
    k = k+1



