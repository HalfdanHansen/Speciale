#Test time
import timeit
from PackagesAndModels.pack import *
from PackagesAndModels.method_functions import *
from PackagesAndModels.train_val_test_CIFAR10 import *
from PackagesAndModels.CIFAR_MODELS import *

time_list = []
n = 300
name_list = ["normal", "CD 3D", "ATCD 3D"]

#normal
mysetup = '''
from PackagesAndModels.train_val_test_CIFAR10 import evaluate_cifar, load_cifar
from numpy import load
from pathlib import Path 
import torchvision.transforms as transforms
import torch
import os
transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

batchsize = 100
trainloader, testloader = load_cifar()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netnormal = torch.load("0905_conv500normalCIFAR10", map_location=torch.device("cpu"))
netnormal.to(device)
netnormal.eval()
'''

mycode = '''
evaluate_cifar(testloader, netnormal).cpu().item()
'''

et = timeit.timeit(setup = mysetup,
                   stmt = mycode,
                   number = n)
et = et/n
time_list.append(et)

print(time_list)

#CD 3D
mysetup = '''
from PackagesAndModels.train_val_test_CIFAR10 import evaluate_cifar, load_cifar
from numpy import load
from pathlib import Path 
import torchvision.transforms as transforms
import torch
import os
transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

batchsize = 100
trainloader, testloader = load_cifar()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = torch.load("0905_conv500PARAFAC3DCIFAR10", map_location=torch.device("cpu"))
net.to(device)
net.eval()
'''

mycode = '''
evaluate_cifar(testloader, net).cpu().item()
'''

et = timeit.timeit(setup = mysetup,
                   stmt = mycode,
                   number = n)
et = et/n
time_list.append(et)

print(time_list)

#ATCD 3D
mysetup = '''
from PackagesAndModels.train_val_test_CIFAR10 import evaluate_cifar, load_cifar
from numpy import load
from pathlib import Path 
import torchvision.transforms as transforms
import torch
import os
transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

batchsize = 100
trainloader, testloader = load_cifar()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = torch.load("0905_conv500ATCD3DCIFAR10", map_location=torch.device("cpu"))
net.to(device)
net.eval()
'''

mycode = '''
evaluate_cifar(testloader, net).cpu().item()
'''

et = timeit.timeit(setup = mysetup,
                   stmt = mycode,
                   number = n)
et = et/n
time_list.append(et)

print(time_list)