
"""
In this file, we build two classes : the MLP and the OurDataset class.
There is also a  function train_model which we use to train our model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


class MLP(nn.Module):
    """
    This class represent our Multilayer Perceptron (MLP).
    As it was ask, we use only one hidden layer
    """
    def __init__(self, neuron_number):
        super(MLP, self).__init__()

        # 28*28 is the size of the image input
        self.fc1 = nn.Linear(28 * 28, neuron_number, bias=True)

        # the output dimension is 10 since we have 10 different class
        self.fc2 = nn.Linear(neuron_number, 10, bias=True)

    def forward(self, x):
        # we flat the image input
        x = x.view(x.size(0), -1)

        # first layer, with relu activation function
        x = self.fc1(x)
        x = F.relu(x)

        # output layer
        x = self.fc2(x)
        return x


class OurDataset(Dataset):
    """
    We use this class to build our data set.
    """
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        label = self.labels[index]
        datapoint = self.data[index]
        # Reshape
        datapoint = datapoint.reshape((28, 28))
        label = label.reshape(1)
        if self.transform:
            # We normalize and put it in a tensor
            datapoint = datapoint/256
            datapoint = self.transform(datapoint)

        return datapoint, torch.tensor([label])

    def __len__(self):
        return len(self.data)


def train_model(model, train_set, validation_set, n_epoch, optimizer, loss_fn):
    """
    We use this function to train our model on the train_set using the optimizer, a loss function loss_fn,
    and n_epoch epochs.
    After, each epoch, we test our model on the validation set.
    We return tr_accuracy, tr_loss, val_accuracy, val_loss, best_model
    tr_accuracy[i], val_accuracy[i] are the accuracy on the train and validation set  at the ith epoch
    tr_loss[i], val_loss[i] are the mean loss on the train and the validation set at the ith epoch
    best_model is the model at epoch i, which has the best accuracy on the validation set
    """
    tr_accuracy = []
    tr_loss = []
    val_accuracy = []
    val_loss = []

    for epoch in range(n_epoch):
        train_accuracy = 0
        validation_accuracy = 0
        train_loss = 0
        validation_loss = 0
        # First we train our model :
        model.train()
        for data, label in train_set:
            optimizer.zero_grad()
            output = model(data.float())

            label = label.squeeze(1)
            label = label.squeeze(1)

            if torch.argmax(output) == label:
                train_accuracy = train_accuracy + 1

            loss = loss_fn(output, label.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_accuracy = train_accuracy / len(train_set.sampler)
        train_loss = train_loss / len(train_set.sampler)

        tr_accuracy.append(train_accuracy)
        tr_loss.append(train_loss)

        # After the training, we evaluate our model on the validation set
        model.eval()
        for data, label in validation_set:
            output = model(data.float())
            label = label.squeeze(1)
            label = label.squeeze(1)
            if torch.argmax(output) == label:
                validation_accuracy = validation_accuracy + 1
            loss = loss_fn(output, label.long())
            validation_loss += loss.item()

        validation_accuracy = validation_accuracy / len(validation_set.sampler)
        validation_loss = validation_loss / len(validation_set.sampler)

        # We search our best model :
        if len(val_accuracy) > 0:
            if validation_accuracy > np.max(val_accuracy):
                best_model = model

        val_accuracy.append(validation_accuracy)
        val_loss.append(validation_loss)

    return tr_accuracy, tr_loss, val_accuracy, val_loss, best_model

# trains models without validation at each epoch
def train_only(model, train_set, n_epoch, optimizer, loss_fn):
    for epoch in range(n_epoch):
        model.train()
        for data, label in train_set:
            optimizer.zero_grad()
            output = model(data.float())

            label = label.squeeze(1)
            label = label.squeeze(1)

            loss = loss_fn(output, label.long())
            loss.backward()
            optimizer.step()

    return model

# Validation or Test of a CNN
def eval_model(model, validation_set, loss_fn):
    validation_accuracy = 0
    validation_loss = 0

    model.eval()
    for data, label in validation_set:
        output = model(data.float())
        label = label.squeeze(1)
        label = label.squeeze(1)
        if torch.argmax(output) == label:
            validation_accuracy = validation_accuracy + 1
        loss = loss_fn(output, label.long())
        validation_loss += loss.item()

    validation_accuracy = validation_accuracy / len(validation_set.sampler)
    validation_loss = validation_loss / len(validation_set.sampler)

    return validation_accuracy, validation_loss
