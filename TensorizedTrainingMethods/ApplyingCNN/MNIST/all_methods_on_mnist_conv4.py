from sklearn.metrics import accuracy_score
from copy import deepcopy

import sys
sys.path.insert(1,'TensorizedTrainingMethods/PackagesAndModels')

from pack import *
from MNIST_CNN_conv4 import *
from train_val_test import * 
from misc_functions import *
from method_functions import *

criterion = nn.CrossEntropyLoss()

convName =  ["conv_1", "conv_2", "conv_3", "conv_4"]
R = 1

batch_size = 100
num_epochs = 50
num_samples_train = x_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

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

    if epoch % 10 == 0:
        print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                epoch+1, losses[-1], train_acc_cur, valid_acc_cur))

epoch = np.arange(len(train_acc))
plt.figure()
plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
plt.legend(['Train Acc', 'Val Acc'])
plt.xlabel('Epochs')
plt.ylabel('Acc')

preds, test_acc_full = evaluate_test(x_test, targets_test, netNormal)
normalvalid_acc = valid_acc
normaltest_acc = accuracy_score(list(targets_test), list(preds.data.numpy()))
print("\nTest set Acc:  %f" % normaltest_acc)

"""Direct 4D Decompose Method

"""

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = []

netD4DD = deepcopy(convNet4)
netD4DD.apply(weight_reset)
optimizerD4DD = optim.Adam(netD4DD.parameters(), lr=0.001)

netD4DD.conv_1 = conv_to_PARAFAC_firsttry(netD4DD.conv_1, R)
netD4DD.conv_2 = conv_to_PARAFAC_firsttry(netD4DD.conv_2, R)
netD4DD.conv_3 = conv_to_PARAFAC_firsttry(netD4DD.conv_3, R)
netD4DD.conv_4 = conv_to_PARAFAC_firsttry(netD4DD.conv_4, R)

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
    
    if epoch % 10 == 0:
        print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                epoch+1, losses[-1], train_acc_cur, valid_acc_cur))
        
epoch = np.arange(len(train_acc))
plt.figure()
plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
plt.legend(['Train Acc', 'Val Acc'])
plt.xlabel('Epochs')
plt.ylabel('Acc')

preds, test_acc_full = evaluate_test(x_test, targets_test, netD4DD)
D4DDvalid_acc = valid_acc
D4DDtest_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))
print("\nTest set Acc:  %f" % D4DDtest_acc)

"""Direct 3D Decompose Method"""

from MNIST_CNN_conv4_3D import *

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
    
    if epoch % 10 == 0:
        print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                epoch+1, losses[-1], train_acc_cur, valid_acc_cur))
        
epoch = np.arange(len(train_acc))
plt.figure()
plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
plt.legend(['Train Acc', 'Val Acc'])
plt.xlabel('Epochs')
plt.ylabel('Acc')

preds, test_acc_full = evaluate_test(x_test, targets_test, net3D)
D3DDvalid_acc = valid_acc
D3DDtest_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))
print("\nTest set Acc:  %f" % D3DDtest_acc)

"""BAF method 4D

"""

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

    if epoch % 10 == 0:
        print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                epoch+1, losses[-1], train_acc_cur, valid_acc_cur))

epoch = np.arange(len(train_acc))
plt.figure()
plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
plt.legend(['Train Acc', 'Val Acc'])
plt.xlabel('Epochs')
plt.ylabel('Acc')

#Evaluate test
preds, test_acc_full = evaluate_test(x_test, targets_test, netBAF4D)
BAF4Dvalid_acc = valid_acc
BAF4Dtest_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))
print("\nTest set Acc:  %f" % BAF4Dtest_acc)

"""BAF method 3D """

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = []

netBAF3D = deepcopy(convNet4)
netBAF3D.apply(weight_reset)
optimizerBAF3D = optim.Adam(netBAF3D.parameters(), lr=0.001)

for epoch in range(num_epochs):
    # Forward -> Backprob -> Update params
    ## Train
    losses = train_net_BAF_3D(losses, num_batches_train, x_train, batch_size, targets_train, netBAF3D, criterion, optimizerBAF3D, convName, R)
    #Evaluate training and validation
    train_targs, train_preds = evaluate_train(num_batches_train, x_train, batch_size, targets_train, netBAF3D)
    val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netBAF3D)
    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)

    if epoch % 5 == 0:
        print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                epoch+1, losses[-1], train_acc_cur, valid_acc_cur))

epoch = np.arange(len(train_acc))
plt.figure()
plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
plt.legend(['Train Acc', 'Val Acc'])
plt.xlabel('Epochs')
plt.ylabel('Acc')

#Evaluate test
preds, test_acc_full = evaluate_test(x_test, targets_test, netBAF3D)
BAF3Dvalid_acc = valid_acc
BAF3Dtest_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))
print("\nTest set Acc:  %f" % BAF3Dtest_acc)

"""ATDC Method 3D

"""

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
    
    if epoch % 10 == 0:
        print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                epoch+1, losses[-1], train_acc_cur, valid_acc_cur))
        
epoch = np.arange(len(train_acc))
plt.figure()
plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
plt.legend(['Train Acc', 'Val Acc'])
plt.xlabel('Epochs')
plt.ylabel('Acc')

### Evaluate test set
preds, test_acc_full = evaluate_test(x_test, targets_test, netATDC)
ATDC3Dvalid_acc = valid_acc
ATDC3Dtest_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))
print("\nTest set Acc:  %f" % ATDC3Dtest_acc)

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
# run training with ATDC step
for epoch in range(num_epochs):
    #Train
    losses = train_net_ATDC_4D(losses, num_batches_train, x_train, batch_size, targets_train, netATDC4D, criterion, optimizerATDC4D, convName, pqtu_convs, alpha)
    ### Evaluate training
    train_targs, train_preds = evaluate_train(num_batches_train, x_train, batch_size, targets_train, netATDC4D)
    ### Evaluate validation
    val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netATDC4D)

    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)
    
    if epoch % 5 == 0:
        print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                epoch+1, losses[-1], train_acc_cur, valid_acc_cur))
        
epoch = np.arange(len(train_acc))
plt.figure()
plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
plt.legend(['Train Acc', 'Val Acc'])
plt.xlabel('Epochs')
plt.ylabel('Acc')

### Evaluate test set
preds, test_acc_full = evaluate_test(x_test, targets_test, netATDC4D)
ATDC4Dvalid_acc = valid_acc
ATDC4Dtest_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))
print("\nTest set Acc:  %f" % ATDC4Dtest_acc)

"""Comparing the Methods"""

import matplotlib.pyplot as plt
epoch = range(50)

parameterFull = 3*1*3*3+6*3*3*3+12*6*3*3+3*12*3*3+7*7*3*10
parameter3D = 3*(1+3+3)+6*(3+3+3)+12*(6+3+3)+3*(12+3+3)+7*7*3*10
parameter4D = (1+3+3+3)+(3+3+3+6)+(6+3+3+12)+(12+3+3+3)+7*7*3*10

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(11,6))

ax1.plot(epoch, normalvalid_acc, 'r-',
         epoch, D3DDvalid_acc, 'b--',
         epoch, D4DDvalid_acc, 'b:',
         epoch, BAF3Dvalid_acc, 'm--',
         epoch, BAF4Dvalid_acc, 'm:',
         epoch, ATDC3Dvalid_acc, 'g--',
         epoch, ATDC4Dvalid_acc, 'g:')

ax2.plot(parameterFull,normaltest_acc, 'ro')
ax2.plot(parameter3D,D3DDtest_acc, 'bs')
ax2.plot(parameter4D,D4DDtest_acc, 'bs')
ax2.plot(parameter3D,BAF3Dtest_acc, 'm*')
ax2.plot(parameter4D,BAF4Dtest_acc, 'm*')
ax2.plot(parameter3D,ATDC3Dtest_acc, 'g^')
ax2.plot(parameter4D,ATDC4Dtest_acc, 'g^')

ax1.legend(['Normal', 'D3DD', 'D4DD','BAF3D','BAF4D','ATDC3D','ATDC4D'], loc = 'lower right')
ax2.legend(['Normal', 'D3DD', 'D4DD','BAF3D','BAF4D','ATDC3D','ATDC4D'], loc = 'lower right')
ax1.set_xlabel('Epochs')
ax1.set_ylim([0.5,1])
ax1.set_ylabel('Validation Accuracy')
ax2.set_ylabel('Test Accuracy')
ax2.set_xlabel('Number of Parameters')
ax2.set_ylim([0.5,1])
plt.suptitle('Training Results For Methods', fontsize=14)
plt.show()

# Model bliver bedre jo færre parametre der bliver brugt
# Måske er gradient landskabet nemmere at finde rundt i når det er 3d
# Håbet er at det ser anderledes ud for et dybere netværk
# Specific instance of training

print(normalvalid_acc)
print(D3DDvalid_acc)
print(D4DDvalid_acc)
print(BAF3Dvalid_acc)
print(BAF4Dvalid_acc)
print(ATDC3Dvalid_acc)
print(ATDC4Dvalid_acc)

print(normaltest_acc)
print(D3DDtest_acc)
print(D4DDtest_acc)
print(BAF3Dtest_acc)
print(BAF4Dtest_acc)
print(ATDC3Dtest_acc)
print(ATDC4Dtest_acc)

normalvalid_acc = [0.842, 0.896, 0.928, 0.936, 0.942, 0.946, 0.946, 0.948, 0.952, 0.956, 0.958, 0.962, 0.962, 0.962, 0.962, 0.96, 0.96, 0.958, 0.96, 0.96, 0.962, 0.96, 0.96, 0.96, 0.96, 0.96, 0.964, 0.962, 0.962, 0.962, 0.962, 0.96, 0.96, 0.96, 0.96, 0.958, 0.958, 0.956, 0.958, 0.958, 0.96, 0.962, 0.964, 0.962, 0.96, 0.96, 0.96, 0.962, 0.96, 0.962]
D3DDvalid_acc = [0.718, 0.856, 0.876, 0.888, 0.9, 0.908, 0.92, 0.918, 0.92, 0.924, 0.924, 0.924, 0.926, 0.926, 0.926, 0.928, 0.934, 0.936, 0.936, 0.94, 0.94, 0.938, 0.938, 0.94, 0.944, 0.944, 0.944, 0.944, 0.944, 0.944, 0.944, 0.944, 0.944, 0.942, 0.946, 0.944, 0.946, 0.944, 0.942, 0.942, 0.942, 0.942, 0.942, 0.942, 0.942, 0.942, 0.944, 0.944, 0.942, 0.944]
D4DDvalid_acc = [0.456, 0.538, 0.568, 0.584, 0.596, 0.606, 0.622, 0.634, 0.636, 0.642, 0.65, 0.656, 0.658, 0.662, 0.664, 0.666, 0.668, 0.67, 0.67, 0.676, 0.674, 0.674, 0.674, 0.676, 0.676, 0.678, 0.678, 0.68, 0.68, 0.68, 0.68, 0.68, 0.678, 0.682, 0.684, 0.686, 0.686, 0.682, 0.682, 0.68, 0.684, 0.682, 0.68, 0.678, 0.678, 0.678, 0.678, 0.68, 0.68, 0.68]
BAF3Dvalid_acc = [0.82, 0.912, 0.924, 0.938, 0.942, 0.944, 0.942, 0.94, 0.942, 0.942, 0.942, 0.942, 0.942, 0.942, 0.94, 0.942, 0.942, 0.944, 0.944, 0.946, 0.948, 0.95, 0.95, 0.952, 0.952, 0.952, 0.95, 0.95, 0.952, 0.952, 0.954, 0.954, 0.954, 0.952, 0.954, 0.952, 0.954, 0.954, 0.954, 0.954, 0.952, 0.954, 0.948, 0.95, 0.948, 0.95, 0.95, 0.948, 0.946, 0.944]
BAF4Dvalid_acc = [0.456, 0.548, 0.568, 0.594, 0.624, 0.686, 0.732, 0.78, 0.81, 0.83, 0.838, 0.846, 0.846, 0.846, 0.858, 0.852, 0.856, 0.852, 0.856, 0.856, 0.86, 0.86, 0.862, 0.862, 0.858, 0.866, 0.866, 0.868, 0.87, 0.872, 0.872, 0.876, 0.876, 0.876, 0.874, 0.878, 0.878, 0.878, 0.88, 0.88, 0.88, 0.88, 0.882, 0.884, 0.884, 0.884, 0.884, 0.886, 0.886, 0.886]
ATDC3Dvalid_acc = [0.748, 0.85, 0.87, 0.884, 0.892, 0.896, 0.91, 0.916, 0.92, 0.92, 0.924, 0.928, 0.92, 0.922, 0.924, 0.922, 0.924, 0.93, 0.932, 0.932, 0.932, 0.93, 0.93, 0.932, 0.932, 0.932, 0.93, 0.932, 0.932, 0.934, 0.934, 0.932, 0.932, 0.93, 0.928, 0.928, 0.93, 0.93, 0.93, 0.93, 0.932, 0.93, 0.93, 0.93, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928]
ATDC4Dvalid_acc = [0.61, 0.778, 0.788, 0.82, 0.84, 0.846, 0.854, 0.862, 0.864, 0.862, 0.862, 0.86, 0.856, 0.858, 0.862, 0.864, 0.862, 0.864, 0.86, 0.864, 0.866, 0.866, 0.868, 0.868, 0.866, 0.868, 0.874, 0.88, 0.882, 0.884, 0.888, 0.886, 0.888, 0.888, 0.886, 0.886, 0.888, 0.888, 0.888, 0.888, 0.886, 0.886, 0.886, 0.888, 0.888, 0.886, 0.886, 0.886, 0.886, 0.888]
normaltest_acc = 0.974
D3DDtest_acc = 0.964
D4DDtest_acc = 0.652
BAF3Dtest_acc = 0.964
BAF4Dtest_acc = 0.878
ATDC3Dtest_acc = 0.952
ATDC4Dtest_acc = 0.92

