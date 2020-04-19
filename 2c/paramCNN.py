'''
Test different Learning rate for the CNN

Output:

Learning rate :
0.001
Mean Loss :
0.06922356978297253
Accuracy :
0.9798
-------------------------------------------------------
Learning rate :
0.0025
Mean Loss :
0.05819047750620352
Accuracy :
0.9825
-------------------------------------------------------
Learning rate :
0.005
Mean Loss :
0.05855046715982454
Accuracy :
0.9833
-------------------------------------------------------
Learning rate :
0.0075
Mean Loss :
0.075207997082443
Accuracy :
0.9821
-------------------------------------------------------
Learning rate :
0.01
Mean Loss :
0.0803603203729205
Accuracy :
0.9845
-------------------------------------------------------
Learning rate :
0.025
Mean Loss :
0.1499974609144947
Accuracy :
0.9782
-------------------------------------------------------
Learning rate :
0.05
Mean Loss :
1.0324599869369107
Accuracy :
0.9694
-------------------------------------------------------
Learning rate :
0.075
Mean Loss :
4.904930742301099
Accuracy :
0.9558
-------------------------------------------------------
Learning rate :
0.1
Mean Loss :
38.082652557859205
Accuracy :
0.9302
-------------------------------------------------------


'''



#try this ? ->  https://github.com/skorch-dev/skorch
import csv
import numpy as np
import utils

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose

import model_task2c
import torch.nn as nn
import torch.optim as optim

import torch


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

#---------------------------------
# Define a CNN - c.f. model_task2c
#---------------------------------
nb_epoch = 10
learnR = 0
cnn = model_task2c.PR_CNN()

#----------------------------
# Optimizer and loss function
#----------------------------
for learnR in [0.001, 0.0025, 0.005 , 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1]:
    print("Learning rate : ")
    print(learnR)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=learnR)

    #----------------
    # Train the model
    #----------------
    _, _, _, _, best_model = utils.train_model(cnn, dataloader_train_set, dataloader_val, nb_epoch, optimizer, loss_fn)

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

    # results
    print("Mean Loss : ")
    print(test_loss)
    print("Accuracy :")
    print(test_accuracy)
    print("-------------------------------------------------------")
