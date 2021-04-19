from sklearn.metrics import accuracy_score
from copy import deepcopy
#from icecream import ic

<<<<<<< HEAD
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
=======
#import os 
#print(os.getcwd())
#from pathlib import Path
#os.chdir(str(Path(os.getcwd()).parents[1]))
#os.chdir(os.getcwd()+'/PackagesAndModels')

from PackagesAndModels.pack import *
from PackagesAndModels.MNIST_MODELS import *
from PackagesAndModels.train_val_test_MNIST import *
from PackagesAndModels.method_functions import *
>>>>>>> refs/remotes/origin/main

train_list = []
test_list = []
valid_list = []
criterion = nn.CrossEntropyLoss()

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

netNormal = deepcopy(convNet500)
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

train_list.append(train_acc)
valid_list.append(normalvalid_acc)
test_list.append(normaltest_acc)


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

train_list.append(train_acc)
valid_list.append(D4DDvalid_acc)
test_list.append(D4DDtest_acc)

"""Direct 3D Decompose Method"""

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = []

convNet5003D.apply(weight_reset)
optimizer3D = optim.Adam(convNet5003D.parameters(), lr=0.001)

for epoch in range(num_epochs):
    cur_loss = 0
    convNet5003D.train()
    losses = train_net(losses, num_batches_train, x_train, batch_size,targets_train, convNet5003D, criterion, optimizer3D)
    train_preds, train_targs = [], []

    train_targs, train_preds = evaluate_train(num_batches_train, x_train, batch_size, targets_train, convNet5003D)
    val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, convNet5003D)

    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)

preds, test_acc_full = evaluate_test(x_test, targets_test, convNet5003D)
D3DDvalid_acc = valid_acc
D3DDtest_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))

train_list.append(train_acc)
valid_list.append(D3DDvalid_acc)
test_list.append(D3DDtest_acc)


"""ATDC Method 3D"""


netATDC = deepcopy(convNet500)
netATDC.apply(weight_reset)
optimizerATDC = optim.Adam(netATDC.parameters(), lr=0.001)

pqt_convs = []

for c in convName:
  convData = eval("netATDC."+c+".weight.data")
  de_layer = []
  for cc in convData:
    temp = tl.decomposition.parafac(tl.tensor(cc), rank = 1)
    de_layer.append(temp[1])
  pqt_convs.append(de_layer)

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = []

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

train_list.append(train_acc)
valid_list.append(ATDC3Dvalid_acc)
test_list.append(ATDC3Dtest_acc)


"""ATDC method 4D"""


netATDC4D = deepcopy(convNet500)
netATDC4D.apply(weight_reset)
optimizerATDC4D = optim.Adam(netATDC4D.parameters(), lr=0.001)

pqtu_convs = []

for c in convName:
  convData = eval("netATDC4D."+c+".weight.data")
  temp = tl.decomposition.parafac(tl.tensor(convData), rank = 1)
  pqtu_convs.append(temp)

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = []

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
    
### Evaluate test set
preds, test_acc_full = evaluate_test(x_test, targets_test, netATDC4D)
ATDC4Dvalid_acc = valid_acc
ATDC4Dtest_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))

train_list.append(train_acc)
valid_list.append(ATDC4Dvalid_acc)
test_list.append(ATDC4Dtest_acc)

save_train = pd.DataFrame(train_list)
save_valid = pd.DataFrame(valid_list)
save_test = pd.DataFrame(test_list)

pd.concat([save_train,save_valid,save_test], axis = 0).to_csv('MNIST_conv500_5methods.csv',index=False,header=False)

