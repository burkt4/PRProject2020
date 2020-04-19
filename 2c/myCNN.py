'''

Basic CNN implementation used to optimize the number of training iteration

Output (+2PNG): 

The mean loss of our best model on the test set is :
0.08220392164790102
The accuracy of our best model on the test set is :
0.9848
'''



import csv
import numpy as np
import utils

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose

import model_task2c
import torch.nn as nn
import torch.optim as optim

import torch

import matplotlib.pyplot as plt


#-------------------------------
# load data - from Felicien work
#-------------------------------
training_set = []
labels_training = []
validation_set = []
labels_validation = []
# read data
with open('../dataset/mnist_train.csv', 'r') as f:
    reader = csv.reader(f)
    line_number = 0
    for line in reader:
        if line_number < 42000:
            x = np.array(line[0])
            labels_training.append(x.astype(np.float))
            y = np.array(line[1:])
            training_set.append(y.astype(np.float))
        else:
            x = np.array(line[0])
            labels_validation.append(x.astype(np.float))
            y = np.array(line[1:])
            validation_set.append(y.astype(np.float))
        line_number = line_number+1
# pytorch data
transforms = Compose([
    ToTensor(),
])
train_set = utils.OurDataset(training_set, labels_training, transforms)
val_set = utils.OurDataset(validation_set, labels_validation, transforms)
dataloader_train_set = DataLoader(train_set, batch_size=1, shuffle=True)
dataloader_val = DataLoader(val_set, batch_size=1, shuffle=True)

print('STEP 1: training set loaded')
#---------------------------------
# Define a CNN - c.f. model_task2c
#---------------------------------
# parameters
nb_epoch = 50
learnR = 0.01

cnn = model_task2c.PR_CNN()

#----------------------------
# Optimizer and loss function
#----------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=learnR)

print('STEP 2: model created')
#----------------
# Train the model
#----------------
tr_accuracy, tr_loss, val_accuracy, val_loss, best_model = utils.train_model(cnn, dataloader_train_set, dataloader_val, nb_epoch, optimizer, loss_fn)

print('STEP 3: model trained')
#---------------
# Test the model
#---------------
# load test set
test_set = []
labels_test = []
with open('../dataset/mnist_test.csv', 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        x = np.array(line[0])
        labels_test.append(x.astype(np.float))
        y = np.array(line[1:])
        test_set.append(y.astype(np.float))

transforms = Compose([
    ToTensor(),
])
test = utils.OurDataset(test_set, labels_test, transforms)
dataloader_test = DataLoader(test, batch_size=1, shuffle=True)
# test model
test_loss = 0
test_accuracy = 0
for data, label in dataloader_test:
    output = best_model(data.float())
    label = label.squeeze(1)
    label = label.squeeze(1)
    if torch.argmax(output) == label:
        test_accuracy = test_accuracy + 1
    loss = loss_fn(output, label.long())
    test_loss += loss.item()

test_accuracy = test_accuracy / len(dataloader_test.sampler)
test_loss = test_loss / len(dataloader_test.sampler)


#------------------
# Model Performance
#------------------
# metrics
print("The mean loss of our best model on the test set is : ")
print(test_loss)
print("The accuracy of our best model on the test set is :")
print(test_accuracy)

iterations = [i+1 for i in range(0, nb_epoch)]
# Accuracy
plt.plot(iterations, tr_accuracy, '--', color="#111111", label="Training accuracy")
plt.plot(iterations, val_accuracy, color="#111111", label="Validation accuracy")
# Create plot
plt.title("Accuracy Curve")
plt.xlabel("Epochs"), plt.ylabel("Accuracy"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig('error_curve.png')
plt.show()

# Loss
plt.plot(iterations, tr_loss, '--', color="#111111", label="Training loss")
plt.plot(iterations, val_loss, color="#111111", label="Validation loss")
# Create plot
plt.title("Loss Curve")
plt.xlabel("Epochs"), plt.ylabel("Loss"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig('loss_curve.png')
plt.show()
