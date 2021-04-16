import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#Import packages and initialize data
from pack import *

import tensorly as tc
import torchvision
import torchvision.transforms as transforms
from method_functions import *
from misc_functions import *
import time

import pandas as pd

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes,affine=False, track_running_stats=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes,affine=False, track_running_stats=False)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine=False, track_running_stats=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        
def conv_3D_block(in_f, out_f, kernelsize,stride,pad = 1):
    return  nn.Sequential(
            nn.Conv2d(in_channels = in_f, out_channels = out_f, kernel_size=1,
                      stride=1, padding=0, bias=False, groups = 1),
            nn.Conv2d(in_channels = out_f, out_channels = out_f, kernel_size=(kernelsize,1),
                      stride=(stride,1), padding=(pad,0), bias=False, groups = out_f),
            nn.Conv2d(in_channels = out_f, out_channels = out_f, kernel_size=(1,kernelsize),
                      stride=(1,stride), padding=(0,pad), bias=False, groups = out_f)
        )

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        skip = stride != 1 or in_planes != self.expansion*planes

        super(BasicBlock3D, self).__init__()
        self.conv1 = conv_3D_block(in_planes,planes,3,1+skip)
        self.bn1 = nn.BatchNorm2d(planes, affine=False, track_running_stats=False)
        self.conv2 = conv_3D_block(planes,planes,3,1)
        self.bn2 = nn.BatchNorm2d(planes, affine=False, track_running_stats=False)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet3D, self).__init__()
        self.in_planes = 64

        self.conv1 = conv_3D_block(3,64,3,1)
        self.bn1 = nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        

def evaluate_cifar(loader,model):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  correct = 0
  total = 0
  with torch.no_grad():
    for data in loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().detach() #item()
  return (correct / total)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

batchsize = 100

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                          shuffle=True, num_workers = 2, pin_memory=True) #numworker =2 and not pin_memory

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                         shuffle=False, num_workers =  2, pin_memory=True) #numworker =2 and not pin_memory

alpha = 0.001

criterion = nn.CrossEntropyLoss()

results_train = []
results_test = []
results_loss = []
epochs = 50

NumModels = 8

for M in range(NumModels):
    for repeats in range(5):
        if M < 4:
            net = ResNet(BasicBlock, [M+1,M+1,M+1,M+1])
        else:
            net = ResNet3D(BasicBlock3D, [M-3,M-3,M-3,M-3])
        net.cuda()

        train_acc = []
        test_acc = []
        losses = []

        optimizer = optim.Adam(net.parameters(), lr=alpha)
        for epoch in range(epochs):
          running_loss = 0
          net.train()

          for i, data in enumerate(trainloader, 0):

            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

          net.eval()
          train_acc.append(evaluate_cifar(trainloader, net).cpu().item())
          test_acc.append(evaluate_cifar(testloader, net).cpu().item())
          losses.append(running_loss)
          
          if epoch > 1 and losses[-1]/losses[-2] > 0.975:
            results_train.append(train_acc)
            results_test.append(test_acc)
            results_loss.append(losses)
            break
            
save_train = pd.DataFrame(results_train)
save_test = pd.DataFrame(results_test)
save_loss = pd.DataFrame(results_loss)
pd.concat([save_train,save_test,save_loss],axis = 0).to_csv('nopassnet_full_and_3D.csv',index=False,header=False)