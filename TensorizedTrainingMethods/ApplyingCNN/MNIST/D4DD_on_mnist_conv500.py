from sklearn.metrics import accuracy_score
from copy import deepcopy
#from icecream import ic

import os 
print(os.getcwd())
from pathlib import Path
os.chdir(str(Path(os.getcwd()).parents[1]))
os.chdir(os.getcwd()+'/PackagesAndModels')

from pack import *
from MNIST_MODELS import *
from train_val_test_MNIST import *
from method_functions import *
from sys import getsizeof

criterion = nn.CrossEntropyLoss()

convName =  ["conv_1", "conv_2", "conv_3", "conv_4"]
R = 1

batch_size = 100
num_epochs = 50
num_samples_train = x_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size


"""Direct 4D Decompose Method"""


train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = [] 

netD4DD = deepcopy(convNet500)
netD4DD.apply(weight_reset)

netD4DD.conv_1 = conv_to_PARAFAC_firsttry(netD4DD.conv_1, R)
netD4DD.conv_2 = conv_to_PARAFAC_firsttry(netD4DD.conv_2, R)
netD4DD.conv_3 = conv_to_PARAFAC_firsttry(netD4DD.conv_3, R)
netD4DD.conv_4 = conv_to_PARAFAC_last_conv_layer(netD4DD.conv_4, R)
optimizerD4DD = optim.Adam(netD4DD.parameters(), lr=0.001)

for epoch in range(num_epochs):
    cur_loss = 0
    netD4DD.train()
    losses = train_net(losses, num_batches_train, x_train, batch_size,targets_train, netD4DD, criterion, optimizerD4DD)
    train_preds, train_targs = [], []

    train_targs, train_preds = evaluate_train(num_batches_train, x_train, batch_size, targets_train, netD4DD)
    val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netD4DD)

    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)


preds, test_acc_full = evaluate_test(x_test, targets_test, netD4DD)
D4DDvalid_acc = valid_acc
D4DDtest_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))

print(train_acc)
