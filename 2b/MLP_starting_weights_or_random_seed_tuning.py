import numpy as np
from sklearn import svm
import csv
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# First, we build our training set
with open('mnist_train.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    matrix = np.array(data, dtype=int)



# For the model selection part, we will not work on all the data set, this could take too much time.
training_length = 20000
training_set = matrix[:training_length, 1:]/255.0
labels_training = matrix[:training_length, 0]
validation_length = 10000
validation_set = matrix[training_length:training_length+validation_length, 1:]/255.0
labels_validation = matrix[training_length:training_length+validation_length, 0]


X,y = training_set, labels_training

# Now, since our data set is ready, we can find the best random weights
#We manually evaluate the accuracy and zero-one-loss after 10 iterations for both the training and the validation set
for seed in [1,26,42,67,123]:
    mlp = MLPClassifier(hidden_layer_sizes=(100),alpha=0.075,max_iter=210,random_state=seed)
    mlp.fit(X, y)
    
    labels_training_pred= mlp.predict(X)
    
    labels_validation_pred= mlp.predict(validation_set)

    print("Seed: {}, accuracy training: {}, accuracy validation: {}".format(seed,accuracy_score(y, labels_training_pred), accuracy_score(labels_validation, labels_validation_pred)))


