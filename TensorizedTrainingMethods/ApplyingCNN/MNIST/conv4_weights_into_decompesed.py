from PackagesAndModels.pack import *
from PackagesAndModels.method_functions import *
from PackagesAndModels.train_val_test_MNIST import *
from PackagesAndModels.MNIST_MODELS import *
import pickle
import timeit

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 100

num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size

# Time for decomposed
time_list = []
n = 10000
#%% 

# ATCD 3D

#Load structure of decomposed network
net3D = deepcopy(convNet43D)
net3D.apply(weight_reset)

convNameDec = []
# getting the names of decomposed model
for name, i in net3D.named_parameters():
  if name.find("bn") == -1:
      name = name.replace(".0.","[0].")
      name = name.replace(".0.","[0].")
      name = name.replace(".1.","[1].")
      name = name.replace(".2.","[2].")
      name = "net3D." + name + ".data"
      convNameDec.append(name)



#Load trained full network with method
netATCD3D = deepcopy(convNet4)
netATCD3D = torch.load("0905_conv4ATDC3DMNIST", map_location=torch.device("cpu"))
netATCD3D.to(device)
netATCD3D.eval()

convName = []
# getting the names of full model
for name, layer in netATCD3D.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        convName.append(name)

# Decomposed weights of method network
pqt_convs = []
for c in convName:
  convData = eval("netATCD3D."+c+".weight.data")
  de_layer = []
  for cc in convData:
      de_layer.append(parafac(tl.tensor(cc), rank = 1)[1])
  pqt_convs.append(de_layer)
              
#Insert decompsed convolution weights into decomposed network structure
for l,layer in enumerate(pqt_convs):
  name1 = eval(convNameDec[l*3+0])
  name2 = eval(convNameDec[l*3+1])
  name3 = eval(convNameDec[l*3+2])

  temp1 = torch.zeros_like(name1)
  temp2 = torch.zeros_like(name2)
  temp3 = torch.zeros_like(name3)
  
  for f,fil in enumerate(layer):
    temp1[f] = torch.tensor(fil[0]).reshape(temp1.shape[1],temp1.shape[2],temp1.shape[3])
    temp2[f] = torch.tensor(fil[1]).reshape(temp2.shape[1],temp2.shape[2],temp2.shape[3])
    temp3[f] = torch.tensor(fil[2]).reshape(temp3.shape[1],temp3.shape[2],temp3.shape[3])
  
  name1[:] = temp1
  name2[:] = temp2
  name3[:] = temp3

#Instert linear weights into decompsed structure
net3D.l_1.weight.data[:] = netATCD3D.l_1.weight.data
#net3D.l_1.bias.data[:] = netATCD3D.l_1.bias.data

val_targsATCD3D, val_predsATCD3D = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netATCD3D)
valid_accATCD3D = accuracy_score(val_targsATCD3D, val_predsATCD3D)

val_targs3D, val_preds3D = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, net3D)
valid_acc3D = accuracy_score(val_targs3D, val_preds3D)

#Test network
print("training accuracy for the ATDC3D model is " + str(valid_accATCD3D))
print("training accuracy for the D3DD   model is " + str(valid_acc3D))

#%%

#Time for weights decompsed network 
mysetup = '''
from numpy import load
from pathlib import Path 
import torch
import os
num_classes = 10
nchannels, rows, cols = 1, 28, 28
p = "mnist.npz"
data = load(p)
x_valid = data["X_valid"][:1000].astype("float32") 
x_valid = x_valid.reshape((-1, nchannels, rows, cols)) 
targets_valid = data["y_valid"][:1000].astype("int32")

from tensorly.decomposition import parafac, tucker
from PackagesAndModels.train_val_test_MNIST import evaluate_validation
from PackagesAndModels.MNIST_MODELS import convNet4, convNet43D
import timeit
from copy import deepcopy
from torch.nn import Conv2d, Linear, CrossEntropyLoss
import torch
from tensorly import tensor as tltensor
from sklearn.metrics import accuracy_score

criterion = CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 100

num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size

def weight_reset(m):
    if isinstance(m, Conv2d) or isinstance(m, Linear):
        m.reset_parameters()

#Load structure of decomposed network
net3D = deepcopy(convNet43D)
net3D.apply(weight_reset)

convNameDec = []
# getting the names of decomposed model
for name, i in net3D.named_parameters():
  if name.find("bn") == -1:
      name = name.replace(".0.","[0].")
      name = name.replace(".0.","[0].")
      name = name.replace(".1.","[1].")
      name = name.replace(".2.","[2].")
      name = "net3D." + name + ".data"
      convNameDec.append(name)

#Load trained full network with method
netATCD3D = deepcopy(convNet4)
netATCD3D = torch.load("0905_conv4ATDC3DMNIST", map_location=torch.device("cpu"))
netATCD3D.to(device)
netATCD3D.eval()

convName = []
# getting the names of full model
for name, layer in netATCD3D.named_modules():
    if isinstance(layer, Conv2d):
        convName.append(name)

# Decomposed weights of method network
pqt_convs = []
for c in convName:
  convData = eval("netATCD3D."+c+".weight.data")
  de_layer = []
  for cc in convData:
      de_layer.append(parafac(tltensor(cc), rank = 1)[1])
  pqt_convs.append(de_layer)
              
#Insert decompsed convolution weights into decomposed network structure
for l,layer in enumerate(pqt_convs):
  name1 = eval(convNameDec[l*3+0])
  name2 = eval(convNameDec[l*3+1])
  name3 = eval(convNameDec[l*3+2])

  temp1 = torch.zeros_like(name1)
  temp2 = torch.zeros_like(name2)
  temp3 = torch.zeros_like(name3)
  
  for f,fil in enumerate(layer):
    temp1[f] = torch.tensor(fil[0]).reshape(temp1.shape[1],temp1.shape[2],temp1.shape[3])
    temp2[f] = torch.tensor(fil[1]).reshape(temp2.shape[1],temp2.shape[2],temp2.shape[3])
    temp3[f] = torch.tensor(fil[2]).reshape(temp3.shape[1],temp3.shape[2],temp3.shape[3])
  
  name1[:] = temp1
  name2[:] = temp2
  name3[:] = temp3

#Instert linear weights into decompsed structure
net3D.l_1.weight.data[:] = netATCD3D.l_1.weight.data
'''

mycode = '''
val_targs3D, val_preds3D = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, net3D)
valid_acc3D = accuracy_score(val_targs3D, val_preds3D)
'''

et = timeit.timeit(setup = mysetup,
                   stmt = mycode,
                   number = n)
et = et/n
time_list.append(et)
#%%

# ATCD 4D rank 1

#Load structure of decomposed network
net4D = deepcopy(convNet44D)
net4D.apply(weight_reset)

#Load trained full network with method
netATCD4D = deepcopy(convNet4)
netATCD4D = torch.load("0905_conv4ATDC4DMNIST", map_location=torch.device("cpu"))
netATCD4D.to(device)
netATCD4D.eval()

convNameDec = []
# getting the names of decomposed model
for name,i in net4D.named_parameters():
  if name.find("bn") == -1:
      name = name.replace(".0.","[0].")
      name = name.replace(".1.","[1].")
      name = name.replace(".2.","[2].")
      name = name.replace(".3.","[3].")
      name = "net4D." + name + ".data"
      convNameDec.append(name)

convName = []
# getting the names of full model
for name, layer in netATCD4D.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        convName.append(name)

# Decomposed weights of method network
pqtu_convs = []
for c in convName:
  convData = eval("netATCD4D."+c+".weight.data")
  pqtu_convs.append(parafac(tl.tensor(convData), rank = 1)[1])
              
#Insert decompsed convolution weights into decomposed network structure
for l,layer in enumerate(pqtu_convs):
  name1 = eval(convNameDec[l*4+0])
  name2 = eval(convNameDec[l*4+1])
  name3 = eval(convNameDec[l*4+2])
  name4 = eval(convNameDec[l*4+3])
  
  temp1 = torch.zeros_like(name1)
  temp2 = torch.zeros_like(name2)
  temp3 = torch.zeros_like(name3)
  temp4 = torch.zeros_like(name4)
  
  temp1 = torch.tensor(layer[1]).reshape(temp1.shape[0],temp1.shape[1],temp1.shape[2],temp1.shape[3])
  temp2 = torch.tensor(layer[2]).reshape(temp2.shape[0],temp2.shape[1],temp2.shape[2],temp2.shape[3])
  temp3 = torch.tensor(layer[3]).reshape(temp3.shape[0],temp3.shape[1],temp3.shape[2],temp3.shape[3])
  temp4 = torch.tensor(layer[0]).reshape(temp4.shape[0],temp4.shape[1],temp4.shape[2],temp4.shape[3])

  name1[:] = temp1
  name2[:] = temp2
  name3[:] = temp3
  name4[:] = temp4

#Instert linear weights into decompsed structure
net4D.l_1.weight.data[:] = netATCD4D.l_1.weight.data
#net3D.l_1.bias.data[:] = netATCD3D.l_1.bias.data

val_targsATCD4D, val_predsATCD4D = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netATCD4D)
valid_accATCD4D = accuracy_score(val_targsATCD4D, val_predsATCD4D)

val_targs4D, val_preds4D = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, net4D)
valid_acc4D = accuracy_score(val_targs4D, val_preds4D)

#Test network
print("training accuracy for the ATDC4D rank 1 model is " + str(valid_accATCD4D))
print("training accuracy for the D4DD rank1  model is " + str(valid_acc4D))

#%%

#Time for weights decompsed network 
mysetup = '''
from numpy import load
from pathlib import Path 
import torch
import os
num_classes = 10
nchannels, rows, cols = 1, 28, 28
p = "mnist.npz"
data = load(p)
x_valid = data["X_valid"][:1000].astype("float32") 
x_valid = x_valid.reshape((-1, nchannels, rows, cols)) 
targets_valid = data["y_valid"][:1000].astype("int32")

from tensorly.decomposition import parafac, tucker
from PackagesAndModels.train_val_test_MNIST import evaluate_validation
from PackagesAndModels.MNIST_MODELS import convNet4, convNet44D
import timeit
from copy import deepcopy
from torch.nn import Conv2d, Linear, CrossEntropyLoss
import torch
from tensorly import tensor as tltensor
from sklearn.metrics import accuracy_score

criterion = CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 100

num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size

def weight_reset(m):
    if isinstance(m, Conv2d) or isinstance(m, Linear):
        m.reset_parameters()

#Load structure of decomposed network
net4D = deepcopy(convNet44D)
net4D.apply(weight_reset)

#Load trained full network with method
netATCD4D = deepcopy(convNet4)
netATCD4D = torch.load("0905_conv4ATDC4DMNIST", map_location=torch.device("cpu"))
netATCD4D.to(device)
netATCD4D.eval()

convNameDec = []
# getting the names of decomposed model
for name,i in net4D.named_parameters():
  if name.find("bn") == -1:
      name = name.replace(".0.","[0].")
      name = name.replace(".1.","[1].")
      name = name.replace(".2.","[2].")
      name = name.replace(".3.","[3].")
      name = "net4D." + name + ".data"
      convNameDec.append(name)

convName = []
# getting the names of full model
for name, layer in netATCD4D.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        convName.append(name)

# Decomposed weights of method network
pqtu_convs = []
for c in convName:
  convData = eval("netATCD4D."+c+".weight.data")
  pqtu_convs.append(parafac(tltensor(convData), rank = 1)[1])
              
#Insert decompsed convolution weights into decomposed network structure
for l,layer in enumerate(pqtu_convs):
  name1 = eval(convNameDec[l*4+0])
  name2 = eval(convNameDec[l*4+1])
  name3 = eval(convNameDec[l*4+2])
  name4 = eval(convNameDec[l*4+3])
  
  temp1 = torch.zeros_like(name1)
  temp2 = torch.zeros_like(name2)
  temp3 = torch.zeros_like(name3)
  temp4 = torch.zeros_like(name4)
  
  temp1 = torch.tensor(layer[1]).reshape(temp1.shape[0],temp1.shape[1],temp1.shape[2],temp1.shape[3])
  temp2 = torch.tensor(layer[2]).reshape(temp2.shape[0],temp2.shape[1],temp2.shape[2],temp2.shape[3])
  temp3 = torch.tensor(layer[3]).reshape(temp3.shape[0],temp3.shape[1],temp3.shape[2],temp3.shape[3])
  temp4 = torch.tensor(layer[0]).reshape(temp4.shape[0],temp4.shape[1],temp4.shape[2],temp4.shape[3])

  name1[:] = temp1
  name2[:] = temp2
  name3[:] = temp3
  name4[:] = temp4

#Instert linear weights into decompsed structure
net4D.l_1.weight.data[:] = netATCD4D.l_1.weight.data
#net3D.l_1.bias.data[:] = netATCD3D.l_1.bias.data
'''

mycode = '''
val_targs4D, val_preds4D = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, net4D)
valid_acc4D = accuracy_score(val_targs4D, val_preds4D)
'''

et = timeit.timeit(setup = mysetup,
                   stmt = mycode,
                   number = n)
et = et/n
time_list.append(et)

#%%

# BAF 3D

#Load structure of decomposed network
net3D = deepcopy(convNet43D)
net3D.apply(weight_reset)

#Load trained full network with method
netBAF3D = deepcopy(convNet4)
netBAF3D = torch.load("0905_conv4BAF3DMNIST", map_location=torch.device("cpu"))
netBAF3D.to(device)
netBAF3D.eval()

convNameDec = []
# getting the names of decomposed model
for name,i in net3D.named_parameters():
  if name.find("bn") == -1:
      name = name.replace(".0.","[0].")
      name = name.replace(".1.","[1].")
      name = name.replace(".2.","[2].")
      name = name.replace(".3.","[3].")
      name = "net3D." + name + ".data"
      convNameDec.append(name)

convName = []
# getting the names of full model
for name, layer in netBAF3D.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        convName.append(name)

# Decomposed weights of method network
pqt_convs = []
for c in convName:
  convData = eval("netBAF3D."+c+".weight.data")
  de_layer = []
  for cc in convData:
      de_layer.append(parafac(tl.tensor(cc), rank = 1)[1])
  pqt_convs.append(de_layer)
              
#Insert decompsed convolution weights into decomposed network structure
for l,layer in enumerate(pqt_convs):
  name1 = eval(convNameDec[l*3+0])
  name2 = eval(convNameDec[l*3+1])
  name3 = eval(convNameDec[l*3+2])

  temp1 = torch.zeros_like(name1)
  temp2 = torch.zeros_like(name2)
  temp3 = torch.zeros_like(name3)
  
  for f,fil in enumerate(layer):
    temp1[f] = torch.tensor(fil[0]).reshape(temp1.shape[1],temp1.shape[2],temp1.shape[3])
    temp2[f] = torch.tensor(fil[1]).reshape(temp2.shape[1],temp2.shape[2],temp2.shape[3])
    temp3[f] = torch.tensor(fil[2]).reshape(temp3.shape[1],temp3.shape[2],temp3.shape[3])
  
  name1[:] = temp1
  name2[:] = temp2
  name3[:] = temp3

#Instert linear weights into decompsed structure
net3D.l_1.weight.data[:] = netBAF3D.l_1.weight.data
#net3D.l_1.bias.data[:] = netATCD3D.l_1.bias.data

val_targsBAF3D, val_predsBAF3D = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netBAF3D)
valid_accBAF3D = accuracy_score(val_targsBAF3D, val_predsBAF3D)

val_targs3D, val_preds3D = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, net3D)
valid_acc3D = accuracy_score(val_targs3D, val_preds3D)

#Test network
print("training accuracy for the BAF3D model is " + str(valid_accBAF3D))
print("training accuracy for the D3DD model is " + str(valid_acc3D))

#%%

#Time for weights decompsed network 
mysetup = '''
from numpy import load
from pathlib import Path 
import torch
import os
num_classes = 10
nchannels, rows, cols = 1, 28, 28
p = "mnist.npz"
data = load(p)
x_valid = data["X_valid"][:1000].astype("float32") 
x_valid = x_valid.reshape((-1, nchannels, rows, cols)) 
targets_valid = data["y_valid"][:1000].astype("int32")

from tensorly.decomposition import parafac, tucker
from PackagesAndModels.train_val_test_MNIST import evaluate_validation
from PackagesAndModels.MNIST_MODELS import convNet4, convNet43D
import timeit
from copy import deepcopy
from torch.nn import Conv2d, Linear, CrossEntropyLoss
import torch
from tensorly import tensor as tltensor
from sklearn.metrics import accuracy_score

criterion = CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 100

num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size

def weight_reset(m):
    if isinstance(m, Conv2d) or isinstance(m, Linear):
        m.reset_parameters()

#Load structure of decomposed network
net3D = deepcopy(convNet43D)
net3D.apply(weight_reset)

#Load trained full network with method
netBAF3D = deepcopy(convNet4)
netBAF3D = torch.load("0905_conv4BAF3DMNIST", map_location=torch.device("cpu"))
netBAF3D.to(device)
netBAF3D.eval()

convNameDec = []
# getting the names of decomposed model
for name,i in net3D.named_parameters():
  if name.find("bn") == -1:
      name = name.replace(".0.","[0].")
      name = name.replace(".1.","[1].")
      name = name.replace(".2.","[2].")
      name = name.replace(".3.","[3].")
      name = "net3D." + name + ".data"
      convNameDec.append(name)

convName = []
# getting the names of full model
for name, layer in netBAF3D.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        convName.append(name)

# Decomposed weights of method network
pqt_convs = []
for c in convName:
  convData = eval("netBAF3D."+c+".weight.data")
  de_layer = []
  for cc in convData:
      de_layer.append(parafac(tltensor(cc), rank = 1)[1])
  pqt_convs.append(de_layer)
              
#Insert decompsed convolution weights into decomposed network structure
for l,layer in enumerate(pqt_convs):
  name1 = eval(convNameDec[l*3+0])
  name2 = eval(convNameDec[l*3+1])
  name3 = eval(convNameDec[l*3+2])

  temp1 = torch.zeros_like(name1)
  temp2 = torch.zeros_like(name2)
  temp3 = torch.zeros_like(name3)
  
  for f,fil in enumerate(layer):
    temp1[f] = torch.tensor(fil[0]).reshape(temp1.shape[1],temp1.shape[2],temp1.shape[3])
    temp2[f] = torch.tensor(fil[1]).reshape(temp2.shape[1],temp2.shape[2],temp2.shape[3])
    temp3[f] = torch.tensor(fil[2]).reshape(temp3.shape[1],temp3.shape[2],temp3.shape[3])
  
  name1[:] = temp1
  name2[:] = temp2
  name3[:] = temp3

#Instert linear weights into decompsed structure
net3D.l_1.weight.data[:] = netBAF3D.l_1.weight.data
#net3D.l_1.bias.data[:] = netATCD3D.l_1.bias.data

'''

mycode = '''
val_targs3D, val_preds3D = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, net3D)
valid_acc3D = accuracy_score(val_targs3D, val_preds3D)
'''

et = timeit.timeit(setup = mysetup,
                   stmt = mycode,
                   number = n)
et = et/n
time_list.append(et)


#%%

# BAF 4D

#Load structure of decomposed network
net4D = deepcopy(convNet44D)
net4D.apply(weight_reset)

#Load trained full network with method
netBAF4D = deepcopy(convNet4)
netBAF4D = torch.load("0905_conv4BAF4DMNIST", map_location=torch.device("cpu"))
netBAF4D.to(device)
netBAF4D.eval()

convNameDec = []
# getting the names of decomposed model
for name,i in net4D.named_parameters():
  if name.find("bn") == -1:
      name = name.replace(".0.","[0].")
      name = name.replace(".1.","[1].")
      name = name.replace(".2.","[2].")
      name = name.replace(".3.","[3].")
      name = "net4D." + name + ".data"
      convNameDec.append(name)

convName = []
# getting the names of full model
for name, layer in netBAF4D.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        convName.append(name)

# Decomposed weights of method network
pqtu_convs = []
for c in convName:
  convData = eval("netBAF4D."+c+".weight.data")
  pqtu_convs.append(parafac(tl.tensor(convData), rank = 1)[1])
              
#Insert decompsed convolution weights into decomposed network structure
for l,layer in enumerate(pqtu_convs):
  name1 = eval(convNameDec[l*4+0])
  name2 = eval(convNameDec[l*4+1])
  name3 = eval(convNameDec[l*4+2])
  name4 = eval(convNameDec[l*4+3])
  
  temp1 = torch.zeros_like(name1)
  temp2 = torch.zeros_like(name2)
  temp3 = torch.zeros_like(name3)
  temp4 = torch.zeros_like(name4)
  
  temp1 = torch.tensor(layer[1]).reshape(temp1.shape[0],temp1.shape[1],temp1.shape[2],temp1.shape[3])
  temp2 = torch.tensor(layer[2]).reshape(temp2.shape[0],temp2.shape[1],temp2.shape[2],temp2.shape[3])
  temp3 = torch.tensor(layer[3]).reshape(temp3.shape[0],temp3.shape[1],temp3.shape[2],temp3.shape[3])
  temp4 = torch.tensor(layer[0]).reshape(temp4.shape[0],temp4.shape[1],temp4.shape[2],temp4.shape[3])

  name1[:] = temp1
  name2[:] = temp2
  name3[:] = temp3
  name4[:] = temp4

#Instert linear weights into decompsed structure
net4D.l_1.weight.data[:] = netBAF4D.l_1.weight.data
#net3D.l_1.bias.data[:] = netATCD3D.l_1.bias.data

val_targsBAF4D, val_predsBAF4D = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netBAF4D)
valid_accBAF4D = accuracy_score(val_targsBAF4D, val_predsBAF4D)

val_targs4D, val_preds4D = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, net4D)
valid_acc4D = accuracy_score(val_targs4D, val_preds4D)

#Test network
print("training accuracy for the ATDC4D rank 1 model is " + str(valid_accBAF4D))
print("training accuracy for the D4DD rank1  model is " + str(valid_acc4D))

#%%

#Time for weights decompsed network 
mysetup = '''
from numpy import load
from pathlib import Path 
import torch
import os
num_classes = 10
nchannels, rows, cols = 1, 28, 28
p = "mnist.npz"
data = load(p)
x_valid = data["X_valid"][:1000].astype("float32") 
x_valid = x_valid.reshape((-1, nchannels, rows, cols)) 
targets_valid = data["y_valid"][:1000].astype("int32")

from tensorly.decomposition import parafac, tucker
from PackagesAndModels.train_val_test_MNIST import evaluate_validation
from PackagesAndModels.MNIST_MODELS import convNet4, convNet44D
import timeit
from copy import deepcopy
from torch.nn import Conv2d, Linear, CrossEntropyLoss
import torch
from tensorly import tensor as tltensor
from sklearn.metrics import accuracy_score

criterion = CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 100

num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size

def weight_reset(m):
    if isinstance(m, Conv2d) or isinstance(m, Linear):
        m.reset_parameters()

#Load structure of decomposed network
net4D = deepcopy(convNet44D)
net4D.apply(weight_reset)

#Load trained full network with method
netBAF4D = deepcopy(convNet4)
netBAF4D = torch.load("0905_conv4BAF4DMNIST", map_location=torch.device("cpu"))
netBAF4D.to(device)
netBAF4D.eval()

convNameDec = []
# getting the names of decomposed model
for name,i in net4D.named_parameters():
  if name.find("bn") == -1:
      name = name.replace(".0.","[0].")
      name = name.replace(".1.","[1].")
      name = name.replace(".2.","[2].")
      name = name.replace(".3.","[3].")
      name = "net4D." + name + ".data"
      convNameDec.append(name)

convName = []
# getting the names of full model
for name, layer in netBAF4D.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        convName.append(name)

# Decomposed weights of method network
pqtu_convs = []
for c in convName:
  convData = eval("netBAF4D."+c+".weight.data")
  pqtu_convs.append(parafac(tltensor(convData), rank = 1)[1])
              
#Insert decompsed convolution weights into decomposed network structure
for l,layer in enumerate(pqtu_convs):
  name1 = eval(convNameDec[l*4+0])
  name2 = eval(convNameDec[l*4+1])
  name3 = eval(convNameDec[l*4+2])
  name4 = eval(convNameDec[l*4+3])
  
  temp1 = torch.zeros_like(name1)
  temp2 = torch.zeros_like(name2)
  temp3 = torch.zeros_like(name3)
  temp4 = torch.zeros_like(name4)
  
  temp1 = torch.tensor(layer[1]).reshape(temp1.shape[0],temp1.shape[1],temp1.shape[2],temp1.shape[3])
  temp2 = torch.tensor(layer[2]).reshape(temp2.shape[0],temp2.shape[1],temp2.shape[2],temp2.shape[3])
  temp3 = torch.tensor(layer[3]).reshape(temp3.shape[0],temp3.shape[1],temp3.shape[2],temp3.shape[3])
  temp4 = torch.tensor(layer[0]).reshape(temp4.shape[0],temp4.shape[1],temp4.shape[2],temp4.shape[3])

  name1[:] = temp1
  name2[:] = temp2
  name3[:] = temp3
  name4[:] = temp4

#Instert linear weights into decompsed structure
net4D.l_1.weight.data[:] = netBAF4D.l_1.weight.data
#net3D.l_1.bias.data[:] = netBAF4D.l_1.bias.data

'''

mycode = '''
val_targs4D, val_preds4D = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, net4D)
valid_acc4D = accuracy_score(val_targs4D, val_preds4D)
'''

et = timeit.timeit(setup = mysetup,
                   stmt = mycode,
                   number = n)
et = et/n
time_list.append(et)
