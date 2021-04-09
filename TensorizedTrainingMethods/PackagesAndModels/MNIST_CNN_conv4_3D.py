#Import packages
import torch.nn as nn
from torch.nn import Linear, Conv2d, MaxPool2d, BatchNorm2d
from torch.nn.functional import relu, softmax
import numpy as np
from MNIST_subset import *

# hyperameters of the model
num_classes = 10
channels = x_train.shape[1]
height = x_train.shape[2]
width = x_train.shape[3]

num_l1 = 10

num_filters_conv = 3 # times 2^(filternumber-1)
kernel_size_conv = 3
padding_conv = 1
stride_conv = 1

def conv_3D_block(in_f, out_f, kernelsize, groups):
    return  nn.Sequential(
            nn.Conv2d(in_channels = in_f, out_channels = out_f, kernel_size=1,
                      stride=1, padding=0, bias=False, groups = 1),
            nn.Conv2d(in_channels = out_f, out_channels = out_f, kernel_size=(kernelsize,1),
                      stride=1, padding=(1,0), bias=False, groups = groups),
            nn.Conv2d(in_channels = out_f, out_channels = out_f, kernel_size=(1,kernelsize),
                      stride=1, padding=(0,1), bias=False, groups = groups)
        )

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_1_3D_block = conv_3D_block(1,3,3,3)
        
        self.bn1 = BatchNorm2d(3)
        
        self.conv_2_3D_block = conv_3D_block(3,6,3,6)
        
        self.bn2 = BatchNorm2d(6)

        self.conv_3_3D_block = conv_3D_block(6,12,3,12)
        
        self.bn3 = BatchNorm2d(12)

        self.conv_4_3D_block = conv_3D_block(12,3,3,3)
        
        self.bn4 = BatchNorm2d(3)
        
        self.l_1 = Linear(in_features = 7*7*3, 
                          out_features = num_l1,
                          bias = False)

        self.maxpool = MaxPool2d(kernel_size = 2,
                                stride = 2)
    
    def forward(self, x): # x.size() = [batch, channel, height, width]
        x = relu(self.conv_1_3D_block(x))     #[x,3,28,28]
        x = self.bn1(x)
        x = self.maxpool(x)                   #[x,3,14,14]
        x = relu(self.conv_2_3D_block(x))     #[x,6,14,14]
        x = self.bn2(x)
        x = self.maxpool(x)                   #[x,6,7,7]
        x = relu(self.conv_3_3D_block(x))     #[x,12,7,7]
        x = self.bn3(x)
        x = relu(self.conv_4_3D_block(x))     #[x,3,7,7]
        x = self.bn4(x)
        x = x.view(-1, 7*7*3)   #[x,3*7*7,1,1] -> [x,10,1,1]
        return softmax(self.l_1(x), dim=1)    #[x,10,1,1]

net3D = Net()

