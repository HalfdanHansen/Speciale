#CNN CIFAR-10

from pack import *

# hyperameters of the model
num_classes = 10
channels = 3
height = 32
width = 32

num_l1 = 10

num_filters_conv = [8,16,32]
kernel_size_conv = 3
padding_conv = 1
stride_conv = 1

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv_1 = Conv2d(in_channels = channels,
                            out_channels = num_filters_conv[0],
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)

        self.bn1 = BatchNorm2d(num_filters_conv[0])

        self.conv_2 = Conv2d(in_channels = num_filters_conv[0],
                            out_channels = num_filters_conv[1],
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)

        self.bn2 = BatchNorm2d(num_filters_conv[1])

        self.conv_3 = Conv2d(in_channels = num_filters_conv[1],
                            out_channels = num_filters_conv[2],
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)
        
        self.bn3 = BatchNorm2d(num_filters_conv[2])

        self.conv_4 = Conv2d(in_channels = num_filters_conv[2],
                            out_channels = num_filters_conv[2],
                            kernel_size = 4,
                            stride = 1,
                            padding = 0,
                            bias = False)
        
        self.bn4 = BatchNorm2d(num_filters_conv[2])

        self.l_1 = Linear(in_features = num_filters_conv[2], 
                          out_features = num_l1,
                          bias = True)
        
        #self.bn5 = BatchNorm1d(num_l1)
        
        self.maxpool = MaxPool2d(kernel_size = 2,
                                stride = 2)

        
    
    def forward(self, x): # x.size() = [batch, channel, height, width]
					      # after application becomes:
        x = relu(self.conv_1(x))              #[x,16,32,32]
        x = self.bn1(x)
        x = self.maxpool(x)                   #[x,16,16,16]
        x = relu(self.conv_2(x))              #[x,32,16,16]
        x = self.bn2(x)
        x = self.maxpool(x)                   #[x,32,8,8]
        x = relu(self.conv_3(x))              #[x,64,8,8]
        x = self.bn3(x)
        x = self.maxpool(x)		              #[x,64,4,4]
        x = relu(self.conv_4(x))              #[x,64,1,1]
        x = self.bn4(x)
        x = x.view(-1, num_filters_conv[2])
        #x = relu(self.l_1(x))
        return softmax(self.l_1(x), dim=1)    #[x,10,1,1]

net = Net()
print(net)