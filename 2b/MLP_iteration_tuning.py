import numpy as np
from sklearn import svm
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import zero_one_loss
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

# Now, since our data set is ready, we can find the best number of iterations

mlp = MLPClassifier(hidden_layer_sizes=(100),alpha=0.075,max_iter=5,warm_start=True)

iterations = list()

training_accuracy_scores = list()
training_loss_scores = list()

validation_accuracy_scores = list()
validation_loss_scores = list()

#We manually evaluate the accuracy and zero-one-loss after each iteration for both the training and the validation set
for i in range(1,50):
    iterations.append(i*5)
    mlp.fit(X, y)
    
    labels_training_pred= mlp.predict(X)
    training_accuracy_scores.append(accuracy_score(y, labels_training_pred))
    training_loss_scores.append(zero_one_loss(labels_training, labels_training_pred))
    
    
    labels_validation_pred= mlp.predict(validation_set)
    validation_accuracy_scores.append(accuracy_score(labels_validation, labels_validation_pred))
    validation_loss_scores.append(zero_one_loss(labels_validation, labels_validation_pred))

# Draw lines
plt.plot(iterations, training_accuracy_scores, '--', color="#111111", label="Training accuracy")
plt.plot(iterations, validation_accuracy_scores, color="#111111", label="Validation accuracy")

# Create plot
plt.title("Accuracy Curve")
plt.xlabel("Epochs"), plt.ylabel("Accuracy"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig('error_curve.png')
plt.show()

# Draw lines
plt.plot(iterations, training_loss_scores, '--', color="#111111", label="Training zero-one-loss")
plt.plot(iterations, validation_loss_scores, color="#111111", label="Validation zero-one-loss")

# Create plot
plt.title("Loss Curve")
plt.xlabel("Epochs"), plt.ylabel("Loss"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig('loss_curve.png')
plt.show()
    
