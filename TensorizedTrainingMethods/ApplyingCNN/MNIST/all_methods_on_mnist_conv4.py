from sklearn.metrics import accuracy_score
from copy import deepcopy

import os 
print(os.getcwd())
import sys 

#import re

#from pathlib import Path
#os.chdir(str(Path(os.getcwd()).parents[2]))
#os.chdir(os.getcwd()+'/PackagesAndModels')
#print(os.getcwd())
from PackagesAndModels.pack import *

#import importlib
#fol = re.sub("/", ".", d)[1:-1]
#importlib.import_module(fol+".pack")

from PackagesAndModels.method_functions import *
from PackagesAndModels.MNIST_MODELS import *
from PackagesAndModels.train_val_test_MNIST import *

criterion = nn.CrossEntropyLoss()

train_list = []
test_list = []
valid_list = []

convName =  ["conv_1", "conv_2", "conv_3", "conv_4"]
R = 1

batch_size = 100
num_epochs = 50
num_samples_train = x_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size

"""Normal Method"""

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = []

netNormal = deepcopy(convNet4)
netNormal.apply(weight_reset)
optimizerNormal = optim.Adam(netNormal.parameters(), lr=0.001)

for epoch in range(num_epochs):
    losses = train_net(losses, num_batches_train, x_train, batch_size,targets_train, netNormal, criterion, optimizerNormal)
    train_targs, train_preds = evaluate_train(num_batches_train, x_train, batch_size, targets_train, netNormal)
    val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netNormal)
    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)

preds, test_acc_full = evaluate_test(x_test, targets_test, netNormal)
normalvalid_acc = valid_acc
normaltest_acc = accuracy_score(list(targets_test), list(preds.data.numpy()))
print("\nTest set Acc:  %f" % normaltest_acc)

train_list.append(train_acc)
valid_list.append(normalvalid_acc)
test_list.append(normaltest_acc)

"""Direct 4D Decompose Method"""

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = []

netD4DD = ConvNet44D()
netD4DD.apply(weight_reset)
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
print("\nTest set Acc:  %f" % D4DDtest_acc)

train_list.append(train_acc)
valid_list.append(D4DDvalid_acc)
test_list.append(D4DDtest_acc)

"""Direct 3D Decompose Method"""

net3D = deepcopy(convNet43D)

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = []

net3D.apply(weight_reset)
optimizer3D = optim.Adam(net3D.parameters(), lr=0.001)

for epoch in range(num_epochs):
    cur_loss = 0
    net3D.train()
    losses = train_net(losses, num_batches_train, x_train, batch_size,targets_train, net3D, criterion, optimizer3D)
    train_preds, train_targs = [], []
    train_targs, train_preds = evaluate_train(num_batches_train, x_train, batch_size, targets_train, net3D)
    val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, net3D)
    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)
    

preds, test_acc_full = evaluate_test(x_test, targets_test, net3D)
D3DDvalid_acc = valid_acc
D3DDtest_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))
print("\nTest set Acc:  %f" % D3DDtest_acc)

train_list.append(train_acc)
valid_list.append(D3DDvalid_acc)
test_list.append(D3DDtest_acc)

"""BAF method 4D"""


train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = []

netBAF4D = deepcopy(convNet4)
netBAF4D.apply(weight_reset)
optimizerBAF4D = optim.Adam(netBAF4D.parameters(), lr=0.001)

for epoch in range(num_epochs):
    losses = train_net_BAF_4D(losses, num_batches_train, x_train, batch_size, targets_train, netBAF4D, criterion, optimizerBAF4D, convName, R)
    train_targs, train_preds = evaluate_train(num_batches_train, x_train, batch_size, targets_train, netBAF4D)
    val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netBAF4D)
    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)

#Evaluate test
preds, test_acc_full = evaluate_test(x_test, targets_test, netBAF4D)
BAF4Dvalid_acc = valid_acc
BAF4Dtest_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))
print("\nTest set Acc:  %f" % BAF4Dtest_acc)

train_list.append(train_acc)
valid_list.append(BAF4Dvalid_acc)
test_list.append(BAF4Dtest_acc)

"""BAF method 3D """


train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = []

netBAF3D = deepcopy(convNet4)
netBAF3D.apply(weight_reset)
optimizerBAF3D = optim.Adam(netBAF3D.parameters(), lr=0.001)

for epoch in range(num_epochs):
    losses = train_net_BAF_3D(losses, num_batches_train, x_train, batch_size, targets_train, netBAF3D, criterion, optimizerBAF3D, convName, R)
    train_targs, train_preds = evaluate_train(num_batches_train, x_train, batch_size, targets_train, netBAF3D)
    val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netBAF3D)
    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)

#Evaluate test
preds, test_acc_full = evaluate_test(x_test, targets_test, netBAF3D)
BAF3Dvalid_acc = valid_acc
BAF3Dtest_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))
print("\nTest set Acc:  %f" % BAF3Dtest_acc)

train_list.append(train_acc)
valid_list.append(BAF3Dvalid_acc)
test_list.append(BAF3Dtest_acc)

"""ATDC Method 3D"""


pqt_convs = []

for c in convName:
  convData = eval("net."+c+".weight.data")
  de_layer = []
  for cc in convData:
    temp = tl.decomposition.parafac(tl.tensor(cc), rank = 1)
    de_layer.append(temp[1])
  pqt_convs.append(de_layer)

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = []

netATDC = deepcopy(convNet4)
netATDC.apply(weight_reset)
optimizerATDC = optim.Adam(netATDC.parameters(), lr=0.001)

alpha = 0.1

for epoch in range(num_epochs):
    losses = train_net_ATDC_3D(losses, num_batches_train, x_train, batch_size, targets_train, netATDC, criterion, optimizerATDC, convName, pqt_convs, alpha)
    train_targs, train_preds = evaluate_train(num_batches_train, x_train, batch_size, targets_train, netATDC)
    val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netATDC)
    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)

### Evaluate test set
preds, test_acc_full = evaluate_test(x_test, targets_test, netATDC)
ATDC3Dvalid_acc = valid_acc
ATDC3Dtest_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))
print("\nTest set Acc:  %f" % ATDC3Dtest_acc)

train_list.append(train_acc)
valid_list.append(ATDC3Dvalid_acc)
test_list.append(ATDC3Dtest_acc)


"""ATDC method 4D"""


pqtu_convs = []

for c in convName:
  convData = eval("net."+c+".weight.data")
  temp = tl.decomposition.parafac(tl.tensor(convData), rank = 1)
  pqtu_convs.append(temp)

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = []

netATDC4D = deepcopy(convNet4)
netATDC4D.apply(weight_reset)
optimizerATDC4D = optim.Adam(netATDC4D.parameters(), lr=0.001)

alpha = 0.1
for epoch in range(num_epochs):
    losses = train_net_ATDC_4D(losses, num_batches_train, x_train, batch_size, targets_train, netATDC4D, criterion, optimizerATDC4D, convName, pqtu_convs, alpha)
    train_targs, train_preds = evaluate_train(num_batches_train, x_train, batch_size, targets_train, netATDC4D)
    val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netATDC4D)
    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)

### Evaluate test set
preds, test_acc_full = evaluate_test(x_test, targets_test, netATDC4D)
ATDC4Dvalid_acc = valid_acc
ATDC4Dtest_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))
print("\nTest set Acc:  %f" % ATDC4Dtest_acc)

train_list.append(train_acc)
valid_list.append(ATDC4Dvalid_acc)
test_list.append(ATDC4Dtest_acc)

save_train = pd.DataFrame(train_list)
save_valid = pd.DataFrame(valid_list)
save_test = pd.DataFrame(test_list)

pd.concat([save_train,save_valid,save_test], axis = 0).to_csv('MNIST_conv4_7methods.csv',index=False,header=False)
