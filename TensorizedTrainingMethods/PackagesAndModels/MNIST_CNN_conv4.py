#Import packages
import torch.nn as nn
from torch.nn import Linear, Conv2d, MaxPool2d, BatchNorm2d
from torch.nn.functional import relu, softmax
import numpy as np
from MNIST_subset import *

'''
data = np.load('mnist.npz')
num_classes = 10
nchannels, rows, cols = 1, 28, 28

x_train = data['X_train'][:10000].astype('float32')
x_train = x_train.reshape((-1, nchannels, rows, cols))
targets_train = data['y_train'][:10000].astype('int32')

x_valid = data['X_valid'][:500].astype('float32')
x_valid = x_valid.reshape((-1, nchannels, rows, cols))
targets_valid = data['y_valid'][:500].astype('int32')

x_test = data['X_test'][:500].astype('float32')
x_test = x_test.reshape((-1, nchannels, rows, cols))
targets_test = data['y_test'][:500].astype('int32')
'''

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

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv_1 = Conv2d(in_channels = channels,
                            out_channels = num_filters_conv*np.power(2,1-1),
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)

        self.bn1 = BatchNorm2d(num_filters_conv*np.power(2,1-1),affine=False,track_running_stats=False)
        
        self.conv_2 = Conv2d(in_channels = num_filters_conv*np.power(2,1-1),
                            out_channels = num_filters_conv*np.power(2,2-1),
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)
        
        self.bn2 = BatchNorm2d(num_filters_conv*np.power(2,2-1),affine=False,track_running_stats=False)

        self.conv_3 = Conv2d(in_channels = num_filters_conv*np.power(2,2-1),
                            out_channels = num_filters_conv*np.power(2,3-1),
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)
        
        self.bn3 = BatchNorm2d(num_filters_conv*np.power(2,3-1),affine=False,track_running_stats=False)

        self.conv_4 = Conv2d(in_channels = num_filters_conv*np.power(2,3-1),
                            out_channels = num_filters_conv*np.power(2,1-1),
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)
        
        self.bn4 = BatchNorm2d(num_filters_conv*np.power(2,1-1),affine=False,track_running_stats=False)
        
        self.l_1 = Linear(in_features = 7*7*3, 
                          out_features = num_l1,
                          bias = False)

        self.maxpool = MaxPool2d(kernel_size = 2,
                                stride = 2)
    
    def forward(self, x): # x.size() = [batch, channel, height, width]
        x = relu(self.conv_1(x))              #[x,3,28,28]
        x = self.bn1(x)
        x = self.maxpool(x)                   #[x,3,14,14]
        x = relu(self.conv_2(x))              #[x,6,14,14]
        x = self.bn2(x)
        x = self.maxpool(x)                   #[x,6,7,7]
        x = relu(self.conv_3(x))              #[x,12,7,7]
        x = self.bn3(x)
        x = relu(self.conv_4(x))              #[x,3,7,7]
        x = self.bn4(x)
        x = x.view(-1, 7*7*3)   #[x,3*7*7,1,1] -> [x,10,1,1]
        return softmax(self.l_1(x), dim=1)    #[x,10,1,1]

net = Net()
print(net)

