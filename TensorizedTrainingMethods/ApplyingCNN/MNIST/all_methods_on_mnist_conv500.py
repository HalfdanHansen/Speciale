from sklearn.metrics import accuracy_score
from copy import deepcopy
#from icecream import ic

import os 
print(os.getcwd())
import sys 

#import re

from pathlib import Path
os.chdir(str(Path(os.getcwd()).parents[1]))
os.chdir(os.getcwd()+'\PackagesAndModels')
print(os.getcwd())
from pack import *
import time
import pickle

#import os 
#print(os.getcwd())
#from pathlib import Path
#os.chdir(str(Path(os.getcwd()).parents[1]))
#os.chdir(os.getcwd()+'/PackagesAndModels')

from pack import *
from MNIST_MODELS import *
from train_val_test_MNIST import *
from method_functions import *

#from PackagesAndModels.pack import *
#from PackagesAndModels.MNIST_MODELS import *
#from PackagesAndModels.train_val_test_MNIST import *
#from PackagesAndModels.method_functions import *

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

"""
#Normal Method


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

torch.save(netNormal, "0905_conv500normalMNIST")

#Direct 4D Decompose Method


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

torch.save(netD4DD, "0905_conv500D4DDMNIST")

#Direct 3D Decompose Method

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

#torch.save(convNet5003D, "0905_conv5003DMNIST")

#ATDC Method 3D


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

### Evaluate test 
preds, test_acc_full = evaluate_test(x_test, targets_test, netATDC)
ATDC3Dvalid_acc = valid_acc
ATDC3Dtest_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))

train_list.append(train_acc)
valid_list.append(ATDC3Dvalid_acc)
test_list.append(ATDC3Dtest_acc)

torch.save(netATDC, "0905_convATDC3DMNIST")

#ATDC method 4D


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

torch.save(netATDC4D, "0905_conv500ATDC4DMNIST")



#Direct 4D Decompose Method rank8
train_list = []
test_list = []
valid_list = []

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = [] 

netD4DD_r8 = deepcopy(convNet5004D_r8)
netD4DD_r8.apply(weight_reset)

#netD4DD.conv_1 = conv_to_PARAFAC_firsttry(netD4DD.conv_1, R)
#netD4DD.conv_2 = conv_to_PARAFAC_firsttry(netD4DD.conv_2, R)
#netD4DD.conv_3 = conv_to_PARAFAC_firsttry(netD4DD.conv_3, R)
#netD4DD.conv_4 = conv_to_PARAFAC_last_conv_layer(netD4DD.conv_4, R)
optimizerD4DD_r8 = optim.Adam(netD4DD_r8.parameters(), lr=0.001)

for epoch in range(num_epochs):
    cur_loss = 0
    netD4DD_r8.train()
    losses = train_net(losses, num_batches_train, x_train, batch_size,targets_train, netD4DD_r8, criterion, optimizerD4DD_r8)
    train_preds, train_targs = [], []

    train_targs, train_preds = evaluate_train(num_batches_train, x_train, batch_size, targets_train, netD4DD_r8)
    val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netD4DD_r8)

    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)

preds, test_acc_full = evaluate_test(x_test, targets_test, netD4DD_r8)
D4DD_r8valid_acc = valid_acc
D4DD_r8test_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))

train_list.append(train_acc)
valid_list.append(D4DD_r8valid_acc)
test_list.append(D4DD_r8test_acc)

torch.save(convNet5003D, "conv5004DMNIST_r8")

save_train = pd.DataFrame(train_list)
save_valid = pd.DataFrame(valid_list)
save_test = pd.DataFrame(test_list)

pd.concat([save_train,save_valid,save_test], axis = 0).to_csv('conv5004DMNIST_r8.csv',index=False,header=False)


#ATDC method 4D rank 8
train_list = []
test_list = []
valid_list = []

def initialize_model_weights_from_PARAFAC_rank_MNIST(convName, net, netname, rank):

  # convName are the names of the layers. In strings
  # netname is the name og the network. In string
  # Make weights from initialization of random weights centered around 0, with std of 0.33.

  pqtu_convs = []
  for k1,c in enumerate(convName):
    convData = eval(netname+"."+c+".weight.data")
    layer =  ([torch.rand(convData.shape[0], rank),
               torch.rand(convData.shape[1], rank),
               torch.rand(convData.shape[2], rank),
               torch.rand(convData.shape[3], rank)])
    #layer =  ([torch.mul(torch.randn(convData.shape[0],rank),0.333).cuda(),
    #           torch.mul(torch.randn(convData.shape[1],rank),0.333).cuda(),
    #           torch.mul(torch.randn(convData.shape[2],rank),0.333).cuda(),
    #           torch.mul(torch.randn(convData.shape[3],rank),0.333).cuda()])

    pqtu_convs.append(layer)
    
  for k1,pqtu in enumerate(pqtu_convs):
      convData = eval(netname+"."+convName[k1]+".weight.data")
      convData[:] = torch.einsum('hsijr->hsij',torch.einsum('hr,sr,ir,jr->hsijr',pqtu[0],pqtu[1],pqtu[2],pqtu[3]))

  return pqtu_convs


netATDC4D_r8 = deepcopy(convNet500)
netATDC4D_r8.apply(weight_reset)
optimizerATDC4D_r8 = optim.Adam(netATDC4D_r8.parameters(), lr=0.001)

pqtu_convs = initialize_model_weights_from_PARAFAC_rank_MNIST(convName, netATDC4D_r8, "netATDC4D_r8", 8)

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = []

alpha = 0.1
# run training with ATDC step
for epoch in range(num_epochs):
    #Train
    losses, pqtu_convs = train_net_ATDC_4D_rank(losses, num_batches_train, x_train, batch_size, targets_train, netATDC4D_r8, criterion, optimizerATDC4D_r8, convName, pqtu_convs, alpha, 8)
    ### Evaluate training
    train_targs, train_preds = evaluate_train(num_batches_train, x_train, batch_size, targets_train, netATDC4D_r8)
    ### Evaluate validation
    val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netATDC4D_r8)

    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)
    
### Evaluate test set
preds, test_acc_full = evaluate_test(x_test, targets_test, netATDC4D_r8)
ATDC4D_r8valid_acc = valid_acc
ATDC4D_r8test_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))

train_list.append(train_acc)
valid_list.append(ATDC4D_r8valid_acc)
test_list.append(ATDC4D_r8test_acc)

torch.save(convNet5003D, "conv5004DATCDMNIST_r8")

save_train = pd.DataFrame(train_list)
save_valid = pd.DataFrame(valid_list)
save_test = pd.DataFrame(test_list)

pqtu_convs1 = decompconvs_to_cpu(pqtu_convs)
pickle.dump(pqtu_convs1, open("conv5004DATCDMNIST_pqtu_r8.p", "wb"))

pd.concat([save_train,save_valid,save_test], axis = 0).to_csv('conv5004DATCDMNIST_r8.csv',index=False,header=False)


#Tucker2 Method

train_list = []
test_list = []
valid_list = []

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = []

netTucker2 = convNet500Tucker2
netTucker2.apply(weight_reset)
optimizerTucker2 = optim.Adam(netTucker2.parameters(), lr=0.001)

for epoch in range(num_epochs):
    cur_loss = 0
    netTucker2.train()
    losses = train_net(losses, num_batches_train, x_train, batch_size,targets_train, netTucker2, criterion, optimizerTucker2)
    train_preds, train_targs = [], []

    train_targs, train_preds = evaluate_train(num_batches_train, x_train, batch_size, targets_train, convNet500Tucker2)
    val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, convNet500Tucker2)

    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)

preds, test_acc_full = evaluate_test(x_test, targets_test, netTucker2)
Tucker2valid_acc = valid_acc
Tucker2test_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))

train_list.append(train_acc)
valid_list.append(Tucker2valid_acc)
test_list.append(Tucker2test_acc)

torch.save(convNet5003D, "conv500Tucker2MNIST_r1")

save_train = pd.DataFrame(train_list)
save_valid = pd.DataFrame(valid_list)
save_test = pd.DataFrame(test_list)

pd.concat([save_train,save_valid,save_test], axis = 0).to_csv('Tucker2conv500MNIST_r1.csv',index=False,header=False)


#Tucker2 rank 8 Method

train_list = []
test_list = []
valid_list = []

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = []

netTucker2_r8 = convNet500Tucker2_r8
netTucker2_r8.apply(weight_reset)
optimizerTucker2_r8 = optim.Adam(netTucker2_r8.parameters(), lr=0.001)

for epoch in range(num_epochs):
    cur_loss = 0
    netTucker2_r8.train()
    losses = train_net(losses, num_batches_train, x_train, batch_size,targets_train, netTucker2_r8, criterion, optimizerTucker2_r8)
    train_preds, train_targs = [], []

    train_targs, train_preds = evaluate_train(num_batches_train, x_train, batch_size, targets_train, convNet500Tucker2_r8)
    val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, convNet500Tucker2_r8)

    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)

preds, test_acc_full = evaluate_test(x_test, targets_test, netTucker2_r8)
Tucker2r_8valid_acc = valid_acc
Tucker2r_8test_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))

train_list.append(train_acc)
valid_list.append(Tucker2r_8valid_acc)
test_list.append(Tucker2r_8test_acc)

torch.save(convNet5003D, "conv500Tucker2MNIST_r8")

save_train = pd.DataFrame(train_list)
save_valid = pd.DataFrame(valid_list)
save_test = pd.DataFrame(test_list)

pd.concat([save_train,save_valid,save_test], axis = 0).to_csv('Tucker2conv500MNIST_r8.csv',index=False,header=False)


#ATDC method Tucker2

train_list = []
test_list = []
valid_list = []

def initialize_model_weights_from_Tucker2_MNIST(convName,net,netname,rank1,rank2,kdims):

  # convName are the names of the layers. In strings
  # netname is the name og the network. In string

  # Make weights from initialization of random weights centered around 0, with std of 0.33.

  utc_convs = []
  for k1,c in enumerate(convName):
    convData = eval(netname+"."+c+".weight.data")
    layer =  ([torch.mul(torch.randn(convData.shape[0],rank2),0.333),     #u
              torch.mul(torch.randn(convData.shape[1],rank1),0.333),      #t
              torch.mul(torch.randn(rank1,rank2,kdims[k1],kdims[k1]),0.333)])             #c
    utc_convs.append(layer)

  for k1,utc in enumerate(utc_convs):
    convData = eval(netname+"."+convName[k1]+".weight.data")
    convData[:] = torch.einsum('hq,sw,wqij->hsij',utc[0],utc[1],utc[2])

  return utc_convs

netAT = deepcopy(convNet500)
netAT.apply(weight_reset)
optimizerAT = optim.Adam(netAT.parameters(), lr=0.001)

utc_convs = []

convName = ['conv_1','conv_2','conv_3','conv_4']
lName = ['l_1']

utc_convs = initialize_model_weights_from_Tucker2_MNIST(convName, netAT, "netAT", 1, 1, [3,3,3,7])

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = []

alpha = 0.1
# run training with ATDC step
for epoch in range(num_epochs):
    #Train
    losses, utc_convs = train_net_ATDC_Tucker2(losses, num_batches_train, x_train, batch_size, targets_train, netAT, criterion, optimizerAT, convName, utc_convs, alpha, 1, 1)
    ### Evaluate training
    train_targs, train_preds = evaluate_train(num_batches_train, x_train, batch_size, targets_train, netAT)
    ### Evaluate validation
    val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netAT)

    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)
    
### Evaluate test set
preds, test_acc_full = evaluate_test(x_test, targets_test, netAT)
ATDCTucker2valid_acc = valid_acc
ATDCTucker2test_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))

train_list.append(train_acc)
valid_list.append(ATDCTucker2valid_acc)
test_list.append(ATDCTucker2test_acc)

torch.save(convNet5003D, "conv500Tucker2ATCDMNIST_r11")

save_train = pd.DataFrame(train_list)
save_valid = pd.DataFrame(valid_list)
save_test = pd.DataFrame(test_list)

utc_convs1 = decompconvs_to_cpu_Tucker(utc_convs)
pickle.dump(utc_convs1, open("conv500Tucker2ATCDMNIST_utc_r11.p", "wb"))

pd.concat([save_train,save_valid,save_test], axis = 0).to_csv('Tucker2ATCDconv500MNIST_r11.csv',index=False,header=False)

"""

#ATDC method Tucker2 rank 8

train_list = []
test_list = []
valid_list = []

def initialize_model_weights_from_Tucker2_MNIST(convName,net,netname,rank1,rank2,kdims):

  # convName are the names of the layers. In strings
  # netname is the name og the network. In string

  # Make weights from initialization of random weights centered around 0, with std of 0.33.

  utc_convs = []
  for k1,c in enumerate(convName):
    convData = eval(netname+"."+c+".weight.data")
    layer =  ([torch.mul(torch.randn(convData.shape[0],rank2),0.333),     #u
              torch.mul(torch.randn(convData.shape[1],rank1),0.333),      #t
              torch.mul(torch.randn(rank1,rank2,kdims[k1],kdims[k1]),0.333)])             #c
    utc_convs.append(layer)

  for k1,utc in enumerate(utc_convs):
    convData = eval(netname+"."+convName[k1]+".weight.data")
    convData[:] = torch.einsum('hq,sw,wqij->hsij',utc[0],utc[1],utc[2])

  return utc_convs

netAT_r8 = deepcopy(convNet500)
netAT_r8.apply(weight_reset)
optimizerAT_r8 = optim.Adam(netAT_r8.parameters(), lr=0.001)

utc_convs = []

convName = ['conv_1','conv_2','conv_3','conv_4']
lName = ['l_1']

utc_convs = initialize_model_weights_from_Tucker2_MNIST(convName, netAT_r8, "netAT_r8", 8, 8, [3,3,3,7])

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = []

alpha = 0.1
# run training with ATDC step
for epoch in range(num_epochs):
    #Train
    losses, utc_convs = train_net_ATDC_Tucker2(losses, num_batches_train, x_train, batch_size, targets_train, netAT_r8, criterion, optimizerAT_r8, convName, utc_convs, alpha, 8, 8)
    ### Evaluate training
    train_targs, train_preds = evaluate_train(num_batches_train, x_train, batch_size, targets_train, netAT_r8)
    ### Evaluate validation
    val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netAT_r8)

    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)
    
### Evaluate test set
preds, test_acc_full = evaluate_test(x_test, targets_test, netAT_r8)
ATDCTucker2_r8valid_acc = valid_acc
ATDCTucker2_r8test_acc = (accuracy_score(list(targets_test), list(preds.data.numpy())))

train_list.append(train_acc)
valid_list.append(ATDCTucker2_r8valid_acc)
test_list.append(ATDCTucker2_r8test_acc)

torch.save(convNet5003D, "conv500Tucker2ATCDMNIST_r88")

save_train = pd.DataFrame(train_list)
save_valid = pd.DataFrame(valid_list)
save_test = pd.DataFrame(test_list)

utc_convs1 = decompconvs_to_cpu_Tucker(utc_convs)
pickle.dump(utc_convs1, open("conv500Tucker2ATCDMNIST_utc_r88.p", "wb"))

pd.concat([save_train,save_valid,save_test], axis = 0).to_csv('Tucker2ATCDconv500MNIST_r88.csv',index=False,header=False)



#pd.concat([save_train,save_valid,save_test], axis = 0).to_csv('0905_MNIST_conv500_5methods.csv',index=False,header=False)