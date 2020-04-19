'''

Random initialisation of CNN and  best model selection based on the validation set

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


np.random.seed(0) #not sure if needed

# parameters
nb_epoch = 20
learnR = 0.01

#----------
# load data
#----------
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

transforms = Compose([
    ToTensor(),
])
train_set = utils.OurDataset(training_set, labels_training, transforms)
val_set = utils.OurDataset(validation_set, labels_validation, transforms)
dataloader_train_set = DataLoader(train_set, batch_size=1, shuffle=True)
dataloader_val = DataLoader(val_set, batch_size=1, shuffle=True)

print('training and validation set loaded')

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

print('test set loaded')


#---------------------------------------
# random initialisation model & training
#---------------------------------------
val_acc = []
val_loss = []
test_acc = []
test_loss = []

seeds = [1,26,42,67,123] # reproduce our results
for seed in seeds:
    print("seed = "+str(seed))

    # make a seeded cnn
    torch.manual_seed(seed)
    cnn = model_task2c.PR_CNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=learnR)

    # Train the model
    best_model = utils.train_only(cnn, dataloader_train_set, nb_epoch, optimizer, loss_fn)

    # validation of the model
    acc, loss = utils.eval_model(best_model, dataloader_val, loss_fn)
    val_acc.append(acc)
    val_loss.append(loss)

    # test the model
    acc, loss = utils.eval_model(best_model, dataloader_test, loss_fn)
    test_acc.append(acc)
    test_loss.append(loss)



#-----------
# Best Model
#-----------

print(val_acc)
print(val_loss)
print(test_acc)
print(test_loss)

val_acc = np.asarray(val_acc)
val_loss = np.asarray(val_loss)

print("Test the model with the highest accuracy on evaluation set :")
model_index = np.argmax(val_acc)
print("seed is : "+str(seeds[model_index]))
print("The mean loss on the test set is : ")
print(test_loss[model_index])
print("The accuracy on the test set is :")
print(test_acc[model_index])

print("Test the model with the lowest mean loss on evaluation set :")
model_index = np.argmin(val_loss)
print("seed is : "+str(seeds[model_index]))
print("The mean loss on the test set is : ")
print(test_loss[model_index])
print("The accuracy on the test set is :")
print(test_acc[model_index])




'''
training and validation set loaded
test set loaded
seed = 1
seed = 26
seed = 42
seed = 67
seed = 123
[0.9805, 0.9827777777777778, 0.9806111111111111, 0.9820555555555556, 0.982]
[0.09317690002776391, 0.0869793067839821, 0.09696220622948509, 0.08706560124398448, 0.09207687696982905]
[0.9844, 0.9851, 0.9841, 0.9854, 0.9857]
[0.06913829392761234, 0.06250857202749184, 0.06746132186693189, 0.06649190447093706, 0.07183752306714461]
Test the model with the highest accuracy on evaluation set :
seed is : 26
The mean loss on the test set is :
0.06250857202749184
The accuracy on the test set is :
'''
