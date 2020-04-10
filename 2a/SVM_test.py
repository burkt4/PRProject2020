"""
Now, since we have find to best model of our different test, we will build a support vector machine using these
 parameters and the all training set.
After that, we will test its accuracy on the all test set.
"""

import numpy as np
from sklearn import svm
import csv
from sklearn.preprocessing import scale
from sklearn import metrics


# First, we build our training set
with open('mnist_train.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    matrix = np.array(data, dtype=int)
    training_set = matrix[:, 1:]
    labels_training = matrix[:, 0]

# Now, we build our test set :
with open('mnist_test.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    matrix = np.array(data, dtype=int)
    test_set = matrix[:, 1:]
    labels_test = matrix[:, 0]

# First we normalize our data :
training_set = training_set/255.0
test_set = test_set/255.0

# And we scale it :
training_set_scaled = scale(training_set)
test_set_scaled = scale(test_set)

# Finally, we can build our best model :
c = 10
gam = 0.001
best_model = svm.SVC(C=c, kernel='rbf', gamma=gam)

# We train our model :
best_model.fit(training_set_scaled, labels_training)

# Then we can compute its accuracy on the test set :
y_pred = best_model.predict(test_set_scaled)
print("accuracy:", metrics.accuracy_score(y_true=labels_test, y_pred=y_pred))


