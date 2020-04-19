import imageio
import glob
import numpy as np
from sklearn import svm
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import zero_one_loss
from sklearn.metrics import accuracy_score

training_set = None
labels_training = None
# First, we build our training set
for i in range(0,10):
    print("/mnist-permutated-png-format/mnist-permutated-png-format/mnist/train/{}/*.png".format(i))
    for image_path in glob.glob("mnist-permutated-png-format/mnist-permutated-png-format/mnist/train/{}/*.png".format(i)):
        im = imageio.imread(image_path)
        im = im.reshape(-1)
        if(training_set is None):
            training_set = im/255.0
            labels_training =[i]
        else:
            training_set = np.vstack ((training_set, im/255.0))
            labels_training = np.concatenate((labels_training, [i]), axis=None)

validation_set = None
labels_validation = None

# Then, we build our validation set
for i in range(0,10):
    print("/mnist-permutated-png-format/mnist-permutated-png-format/mnist/val/{}/*.png".format(i))
    for image_path in glob.glob("mnist-permutated-png-format/mnist-permutated-png-format/mnist/val/{}/*.png".format(i)):
        im = imageio.imread(image_path)
        im = im.reshape(-1)
        if(validation_set is None):
            validation_set = im/255.0
            labels_validation =[i]
        else:
            validation_set = np.vstack ((validation_set, im/255.0))
            labels_validation = np.concatenate((labels_validation, [i]), axis=None)


X,y = training_set, labels_training

# Now, since our data set is ready, we can find the best number of iterations

mlp = MLPClassifier(hidden_layer_sizes=(100),alpha=0.05,max_iter=5,random_state=1,warm_start=True)

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
    
