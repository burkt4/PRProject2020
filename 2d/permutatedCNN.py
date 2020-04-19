'''

Basic CNN running on the permutated dataset

'''



from PIL import Image
import numpy as np
import os, os.path, time

import utils

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose

import model_task2c
import torch.nn as nn
import torch.optim as optim

import torch

import matplotlib.pyplot as plt


#-------------------------
# Load train/val/test DATA
#-------------------------
source = '../dataset/mnist-permutated-png-format/mnist/'
folder = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

training_set = []
labels_training = []
validation_set = []
labels_validation = []
test_set = []
labels_test = []

# load training set
t = 'train'
for f in folder:
    dir= source+t+'/'+str(f)
    for filename in os.listdir(dir):
        if filename.endswith(".png"):
            picture = Image.open(dir+'/'+filename).convert('L')

            x = np.array(f)
            labels_training.append(x.astype(np.float))
            y = np.array(picture.getdata())
            training_set.append(y.astype(np.float))
# load validation valildation set
t = 'val'
for f in folder:
    dir= source+t+'/'+str(f)
    for filename in os.listdir(dir):
        if filename.endswith(".png"):
            picture = Image.open(dir+'/'+filename).convert('L')

            x = np.array(f)
            labels_validation.append(x.astype(np.float))
            y = np.array(picture.getdata())
            validation_set.append(y.astype(np.float))
# load test set
t = 'test'
for f in folder:
    dir= source+t+'/'+str(f)
    for filename in os.listdir(dir):
        if filename.endswith(".png"):
            picture = Image.open(dir+'/'+filename).convert('L')

            x = np.array(f)
            labels_test.append(x.astype(np.float))
            y = np.array(picture.getdata())
            test_set.append(y.astype(np.float))

# pytorch data
transforms = Compose([
    ToTensor(),
])
train_set = utils.OurDataset(training_set, labels_training, transforms)
val_set = utils.OurDataset(validation_set, labels_validation, transforms)
dataloader_train_set = DataLoader(train_set, batch_size=1, shuffle=True)
dataloader_val = DataLoader(val_set, batch_size=1, shuffle=True)
test = utils.OurDataset(test_set, labels_test, transforms)
dataloader_test = DataLoader(test, batch_size=1, shuffle=True)


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
