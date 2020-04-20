"""
In this file, we will measure the performance of two different support vector machine
The first one will use a linear kernel and the second one a gaussian kernel
To measure the differences, we use 5-fold cross-validation approach
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
training_set = training_set[:training_length, :]
labels_training = labels_training[:training_length]

# First we normalize our data :
training_set = training_set/255.0

# And we scale it :
training_set_scaled = scale(training_set)

# First we build a support vector machine with a linear kernel
model_linear = svm.SVC(kernel='linear')
scores_linear = cross_val_score(model_linear, training_set_scaled, labels_training, cv=5)
print("The cross validation score of the model_linear is :")
print(scores_linear)
print("Which mean is :")
print(np.mean(scores_linear))

# Then we build a support vector machine with a gaussian kernel
model_gaussian = svm.SVC(kernel='rbf')
scores_gaussian = cross_val_score(model_gaussian, training_set_scaled, labels_training, cv=5)
print("The cross validation score of the model_gaussian is :")
print(scores_gaussian)
print("Which mean is :")
print(np.mean(scores_gaussian))


