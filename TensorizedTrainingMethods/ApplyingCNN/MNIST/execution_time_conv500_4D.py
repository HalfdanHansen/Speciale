#Test time
import timeit
from PackagesAndModels.pack import *
from PackagesAndModels.method_functions import *
from PackagesAndModels.train_val_test_CIFAR10 import *
from PackagesAndModels.CIFAR_MODELS import *

time_list = []
n = 10000
names = ["CD 4D R1", "ATCD 4D R1", "CD 4D R8", "ATCD 4D R8"]

#CD R1
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
net = torch.load("0905_conv500PARAFAC4DCIFAR10_rank1", map_location=torch.device("cpu"))
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

#CD R8
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
net = torch.load("0905_conv500PARAFAC4DCIFAR10_rank8", map_location=torch.device("cpu"))
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

#ATCD R1
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
net = torch.load("0905_conv500ATCD4DCIFAR10", map_location=torch.device("cpu"))
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

#ATCD R8
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
net = torch.load("0905_conv500ATCD4DCIFAR10_rank8", map_location=torch.device("cpu"))
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
