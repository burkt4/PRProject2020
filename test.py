"""
In this file, we build a MLP with the best parameters we have found
and we test it on the test set.
"""
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose
import utils

# First, we build our training and validation set
# We will put about 70% of the data in the training set and 30% in the validation set
training_set = []
labels_training = []
validation_set = []
labels_validation = []

with open('mnist_train.csv', 'r') as f:
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


# Now, we build our test set :
test_set = []
labels_test = []

with open('mnist_test.csv', 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        x = np.array(line[0])
        labels_test.append(x.astype(np.float))
        y = np.array(line[1:])
        test_set.append(y.astype(np.float))


transforms = Compose([
    ToTensor(),
])
train_set = utils.OurDataset(training_set, labels_training, transforms)
val_set = utils.OurDataset(validation_set, labels_validation, transforms)
test = utils.OurDataset(test_set, labels_test, transforms)

dataloader_train_set = DataLoader(train_set, batch_size=1, shuffle=True)
dataloader_val = DataLoader(val_set, batch_size=1, shuffle=True)
dataloader_test = DataLoader(val_set, batch_size=1, shuffle=True)

# After that, We define the different of our model and our learning approach:
loss_fn = nn.CrossEntropyLoss()
lr = 0.01

neuron_number = 100
n_epoch = 20

model = utils.MLP(neuron_number)
optimizer = torch.optim.SGD(model.parameters(), lr)

# We train our model :
_, _, _, _, best_model = utils.train_model(model, dataloader_train_set, dataloader_val, n_epoch, optimizer, loss_fn)

# Now, since we have selected our best model, we can test it on the test set
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

# Finally, we can print the result :
print("The mean loss of our best model on the test set is : ")
print(test_loss)
print("The accuracy of our best model on the test set is :")
print(test_accuracy)


"""
Output : 
The mean loss of our best model on the test set is : 
0.11104392406151632
The accuracy of our best model on the test set is :
0.9757777777777777

The mean loss of our best model on the test set is : 
0.1082682726588212
The accuracy of our best model on the test set is :
0.9761111111111112

The mean loss of our best model on the test set is : 
0.1102800866415572
The accuracy of our best model on the test set is :

The mean loss of our best model on the test set is : 
0.10571410591522164
The accuracy of our best model on the test set is :
0.9777222222222223

The mean loss of our best model on the test set is : 
0.11369273198088255
The accuracy of our best model on the test set is :
0.9764444444444444

"""
