"""
In this file, we try to find a good number of epoch.
So, we build a model using 40 epochs and we plot the accuracy, loss
on train and on validation set to see what happens
"""

import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose
import utils
import matplotlib.pyplot as plt

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

transforms = Compose([
    ToTensor(),
])
train_set = utils.OurDataset(training_set, labels_training, transforms)
val_set = utils.OurDataset(validation_set, labels_validation, transforms)

dataloader_train_set = DataLoader(train_set, batch_size=1, shuffle=True)
dataloader_val = DataLoader(val_set, batch_size=1, shuffle=True)

# Then, we define the different parameters of our model and our learning approach:
loss_fn = nn.CrossEntropyLoss()
lr = 0.01
# We use 100 neuron since it provide the best result
neuron_number = 100
n_epoch = 40

# We build our model :
model = utils.MLP(neuron_number)
optimizer = torch.optim.SGD(model.parameters(), lr)

# We train our model :
tr_accuracy, tr_loss, val_accuracy, val_loss, _ = utils.train_model(model, dataloader_train_set, dataloader_val,
                                                                    n_epoch, optimizer, loss_fn)

# Finally, we can plot the result :
x = np.arange(n_epoch)
x = x+1

plt.figure()
plt.plot(x, tr_loss)
plt.plot(x, val_loss)

print("tr_loss :")
print(tr_loss)
print("val_loss :")
print(val_loss)
plt.xlabel('number of epoch')
plt.ylabel('loss value')
plt.legend(['loss on the training set', 'loss on validation set'])

plt.figure()
plt.plot(x, tr_accuracy)
plt.plot(x, val_accuracy)
print("tr_accuracy : ")
print(tr_accuracy)
print("Best train accuracy :")
print(max(tr_accuracy))
print("train accuracy mean :")
print(np.mean(tr_accuracy))

print("val_accuracy :")
print(val_accuracy)
print("validation accuracy mean :")
print(np.mean(val_accuracy))
print("Best validation accuracy :")
print(max(val_accuracy))
print("Which correspond to the iteration {}".format(np.argmax(val_accuracy)+1))
plt.xlabel('number of epoch')
plt.ylabel('accuracy')
plt.legend(['accuracy on the training set', 'accuracy on validation set'])
plt.show()


"""
Output :

tr_loss :
[0.2780269703739044, 0.12273310671593676, 0.08851577580299809, 0.06617448021352736, 0.05154866552101272, 0.04066225809482589, 0.03405010727210257, 0.026472800666117857, 0.019527461504671856, 0.014326191573933951, 0.011306996839783742, 0.008209248835952365, 0.0055219377269309314, 0.0034496510303005383, 0.0023878572890147325, 0.0017695402368312816, 0.001498971608317417, 0.0012915504963697997, 0.0010634989483489074, 0.0009966306846754037, 0.0009034156703713101, 0.0008223916663135454, 0.0007862221081866663, 0.0007218810850411675, 0.0006856533413693677, 0.0006457197691633077, 0.0006120815432746562, 0.0005839477392411403, 0.000558308089553898, 0.0005338892528191305, 0.000511455029599638, 0.0004894593396951629, 0.0004714006087870764, 0.00045356332591351155, 0.00043614489931835076, 0.0004221810541335529, 0.0004063936730543645, 0.0003937516619531517, 0.00038230049295406014, 0.00037012517657241854]
val_loss :
[0.1600061562535429, 0.12912327824932515, 0.11638644112601426, 0.10632333447904829, 0.11647794539635457, 0.11017238779435823, 0.13952395157409525, 0.12362289506126235, 0.10644134564099685, 0.11880811399425839, 0.10823437590529672, 0.11370858869987054, 0.11029359777983667, 0.11156014250658908, 0.11142270472846755, 0.11435795933307571, 0.11105538009378291, 0.11287316940735981, 0.11426379703614592, 0.11549161468893183, 0.1154527189238179, 0.115854913449685, 0.1169828126937738, 0.11706863580679579, 0.11761904217172141, 0.11866578113278552, 0.11883305087906514, 0.1197500858644037, 0.12038150217098734, 0.120542648709957, 0.12112548132054872, 0.12164818771654604, 0.12191141879314053, 0.12261676219129622, 0.12218376085179251, 0.12315344769499911, 0.12344539422553504, 0.12359544767046436, 0.12477783455910003, 0.12448920724581886]
tr_accuracy : 
[0.9166904761904762, 0.9632380952380952, 0.973047619047619, 0.9793333333333333, 0.9834047619047619, 0.9871428571428571, 0.9888809523809524, 0.9916666666666667, 0.9942142857142857, 0.9960476190476191, 0.9972142857142857, 0.9979523809523809, 0.9989523809523809, 0.9995952380952381, 0.9998095238095238, 0.9998809523809524, 0.9999285714285714, 0.9999761904761905, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Best train accuracy :
1.0
train accuracy mean :
0.9941744047619048
val_accuracy :
[0.9523333333333334, 0.9612222222222222, 0.9662222222222222, 0.9699444444444445, 0.9691666666666666, 0.9697222222222223, 0.9616111111111111, 0.9684444444444444, 0.9748888888888889, 0.9715555555555555, 0.974, 0.9728888888888889, 0.9756666666666667, 0.9757777777777777, 0.9763333333333334, 0.9763333333333334, 0.9767777777777777, 0.9770555555555556, 0.9771111111111112, 0.9771111111111112, 0.9770555555555556, 0.9768333333333333, 0.9768888888888889, 0.9768888888888889, 0.9771111111111112, 0.9771666666666666, 0.9773333333333334, 0.9773333333333334, 0.9772222222222222, 0.9770555555555556, 0.9769444444444444, 0.9770555555555556, 0.9770555555555556, 0.9768888888888889, 0.9771111111111112, 0.9768888888888889, 0.9770555555555556, 0.9771666666666666, 0.9774444444444444, 0.9774444444444444]
validation accuracy mean :
0.9741527777777778
Best validation accuracy :
0.9774444444444444
Which correspond to the iteration 39

"""