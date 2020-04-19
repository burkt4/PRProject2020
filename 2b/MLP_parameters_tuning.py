"""
In this file, we try to optimize the parameters of a multi layer perceptron
"""

import numpy as np
from sklearn import svm
import csv
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


# First, we build our training set
with open('mnist_train.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    matrix = np.array(data, dtype=int)
    training_set = matrix[:, 1:]
    labels_training = matrix[:, 0]


# For the model selection part, we will not work on all the data set, this could take too much time.
training_length = 40200
training_set = training_set[:training_length, :]
labels_training = labels_training[:training_length]
training_length = 19800


# First we normalize our data :
training_set = training_set/255.0


# And we scale it :
training_set_scaled = scale(training_set)

X,y = training_set_scaled, labels_training

# Now, since our data set is ready, we can find the best parameters for our model
# For the number of neurons in the hidden layer we test 10,20,40,60,80,100
# For learning rate we test 0.001,0.0025,0.005,0.0075, 0.01,0.025,0.05,0.075, 0.1


mlp = MLPClassifier(max_iter=150)

parameter_space = {
    'hidden_layer_sizes': [(10),(20),(40),(60),(80),(100)],
    'alpha': [0.001,0.0025,0.005,0.0075, 0.01,0.025,0.05,0.075, 0.1]
}

gsCV = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
gsCV.fit(X, y)

print('Best parameters found:\n', gsCV.best_params_)
