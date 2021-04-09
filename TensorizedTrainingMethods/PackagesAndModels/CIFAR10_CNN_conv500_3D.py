#Import packages
import torch.nn as nn
from torch.nn import Linear, Conv2d, MaxPool2d, BatchNorm2d
from torch.nn.functional import relu, softmax
import numpy as np
from MNIST_subset import *

from pack import *

# hyperameters of the model
num_classes = 10
channels = x_train.shape[1]
height = x_train.shape[2]
width = x_train.shape[3]

num_l1 = 10

depth = [32,64,128,64]
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

        self.conv_1_3D_block = conv_3D_block(3,depth[0],3,depth[0])
        
        self.bn1 = BatchNorm2d(depth[0],affine=False,track_running_stats=False)
        
        self.conv_2_3D_block = conv_3D_block(depth[0],depth[1],3,depth[1])
        
        self.bn2 = BatchNorm2d(depth[1],affine=False,track_running_stats=False)

        self.conv_3_3D_block = conv_3D_block(depth[1],depth[2],3,depth[2])
        
        self.bn3 = BatchNorm2d(depth[2],affine=False,track_running_stats=False)

        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels = depth[2], out_channels = depth[3], kernel_size=1,
                      stride=1, padding=0, bias=False, groups = 1),
            nn.Conv2d(in_channels = depth[3], out_channels = depth[3], kernel_size=(4,1),
                      stride=1, padding=0, bias=False, groups = depth[3]),
            nn.Conv2d(in_channels = depth[3], out_channels = depth[3], kernel_size=(1,4),
                      stride=1, padding=0, bias=False, groups = depth[3])
        )

        self.bn4 = BatchNorm2d(depth[3],affine=False,track_running_stats=False)
        
        self.l_1 = Linear(in_features = depth[3], 
                          out_features = num_l1,
                          bias = False)

        self.maxpool = MaxPool2d(kernel_size = 2,
                                stride = 2)
    
    def forward(self, x): # x.size() = [batch, channel, height, width]
        x = relu(self.conv_1_3D_block(x))     #[x,32,32,32]
        x = self.bn1(x)
        x = self.maxpool(x)                   #[x,32,16,16]
        x = relu(self.conv_2_3D_block(x))     #[x,64,16,16]
        x = self.bn2(x)
        x = self.maxpool(x)                   #[x,64,8,8]
        x = relu(self.conv_3_3D_block(x))     #[x,128,8,8]
        x = self.bn3(x)
        x = self.maxpool(x)                   #[x,128,4,4]
        x = relu(self.conv_4(x))              #[x,64,1,1]
        x = self.bn4(x)
        x = x.view(-1, depth[3])   #[x,64,1,1] -> [x,10,1,1]
        return softmax(self.l_1(x), dim=1)    #[x,10,1,1]

net = Net()
print(net)