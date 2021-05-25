#Test time
import timeit
from PackagesAndModels.pack import *
from PackagesAndModels.method_functions import *
from PackagesAndModels.train_val_test_MNIST import *
from MNIST_MODELS import *

time_list = []
n = 10000
name_list = ["normal", "CD 3D", "ATCD 3D", "CD 4D", "ATCD 4D"]

#normal
mysetup = '''
from PackagesAndModels.train_val_test_MNIST import evaluate_validation
from numpy import load
from pathlib import Path 
import torch
import os
from sklearn.metrics import accuracy_score
num_classes = 10
nchannels, rows, cols = 1, 28, 28
p = "mnist.npz"
data = load(p)
x_valid = data["X_valid"][:1000].astype("float32") 
x_valid = x_valid.reshape((-1, nchannels, rows, cols)) 
targets_valid = data["y_valid"][:1000].astype("int32")
batch_size = 100
num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size
netnormal = torch.load("0905_conv500normalMNIST", map_location=torch.device("cpu"))
netnormal.eval()
'''

mycode = '''
val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netnormal)
valid_acc = accuracy_score(val_targs, val_preds)
'''

et = timeit.timeit(setup = mysetup,
                   stmt = mycode,
                   number = n)
et = et/n
time_list.append(et)

#CD3D
mysetup = '''
from PackagesAndModels.train_val_test_MNIST import evaluate_validation
from numpy import load
from pathlib import Path 
import torch
import os
from sklearn.metrics import accuracy_score
num_classes = 10
nchannels, rows, cols = 1, 28, 28
p = "mnist.npz"
data = load(p)
x_valid = data["X_valid"][:1000].astype("float32") 
x_valid = x_valid.reshape((-1, nchannels, rows, cols)) 
targets_valid = data["y_valid"][:1000].astype("int32")
batch_size = 100
num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size
netCD3D = torch.load("0905_conv5003DMNIST", map_location=torch.device("cpu"))
netCD3D.eval()
'''

mycode = '''
val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netCD3D)
valid_acc = accuracy_score(val_targs, val_preds)
'''

et = timeit.timeit(setup = mysetup,
                   stmt = mycode,
                   number = n)
et = et/n
time_list.append(et)

#ATCD3D
mysetup = '''
from PackagesAndModels.train_val_test_MNIST import evaluate_validation
from numpy import load
from pathlib import Path 
import torch
import os
from sklearn.metrics import accuracy_score
num_classes = 10
nchannels, rows, cols = 1, 28, 28
p = "mnist.npz"
data = load(p)
x_valid = data["X_valid"][:1000].astype("float32") 
x_valid = x_valid.reshape((-1, nchannels, rows, cols)) 
targets_valid = data["y_valid"][:1000].astype("int32")
batch_size = 100
num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size
netATCD3D = torch.load("0905_conv500ATDC3DMNIST", map_location=torch.device("cpu"))
netATCD3D.eval()
'''

mycode = '''
val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netATCD3D)
valid_acc = accuracy_score(val_targs, val_preds)
'''

et = timeit.timeit(setup = mysetup,
                   stmt = mycode,
                   number = n)
et = et/n
time_list.append(et)

#CD4D
mysetup = '''
from PackagesAndModels.train_val_test_MNIST import evaluate_validation
from numpy import load
from pathlib import Path 
import torch
import os
from sklearn.metrics import accuracy_score
num_classes = 10
nchannels, rows, cols = 1, 28, 28
p = "mnist.npz"
data = load(p)
x_valid = data["X_valid"][:1000].astype("float32") 
x_valid = x_valid.reshape((-1, nchannels, rows, cols)) 
targets_valid = data["y_valid"][:1000].astype("int32")
batch_size = 100
num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size
netCD4D = torch.load("0905_conv4D4DDMNIST", map_location=torch.device("cpu"))
netCD4D.eval()
'''

mycode = '''
val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netCD4D)
valid_acc = accuracy_score(val_targs, val_preds)
'''

et = timeit.timeit(setup = mysetup,
                   stmt = mycode,
                   number = n)
et = et/n
time_list.append(et)

#ATCD4D
mysetup = '''
from PackagesAndModels.train_val_test_MNIST import evaluate_validation
from numpy import load
from pathlib import Path 
import torch
import os
from sklearn.metrics import accuracy_score
num_classes = 10
nchannels, rows, cols = 1, 28, 28
p = "mnist.npz"
data = load(p)
x_valid = data["X_valid"][:1000].astype("float32") 
x_valid = x_valid.reshape((-1, nchannels, rows, cols)) 
targets_valid = data["y_valid"][:1000].astype("int32")
batch_size = 100
num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size
netATCD4D = torch.load("0905_conv500ATDC4DMNIST", map_location=torch.device("cpu"))
netATCD4D.eval()
'''

mycode = '''
val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, netATCD4D)
valid_acc = accuracy_score(val_targs, val_preds)
'''

et = timeit.timeit(setup = mysetup,
                   stmt = mycode,
                   number = n)
et = et/n
time_list.append(et)

print(name_list)
print(time_list)