#from pack import *

num_classes = 10
channels = 3
height = 32
width = 32

num_l1 = 10

depth = [32,64,128,64]
kernel_size_conv = 3
padding_conv = 1
stride_conv = 1

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv_1 = Conv2d(in_channels = channels,
                            out_channels = depth[0],
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)

        self.bn1 = BatchNorm2d(depth[0],affine=False, track_running_stats=False)

        self.conv_2 = Conv2d(in_channels = depth[0],
                            out_channels = depth[1],
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)

        self.bn2 = BatchNorm2d(depth[1],affine=False, track_running_stats=False)

        self.conv_3 = Conv2d(in_channels = depth[1],
                            out_channels = depth[2],
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)
        
        self.bn3 = BatchNorm2d(depth[2],affine=False, track_running_stats=False)

        self.conv_4 = Conv2d(in_channels = depth[2],
                            out_channels = depth[3],
                            kernel_size = 4,
                            stride = 1,
                            padding = 0,
                            bias = False)
        
        self.bn4 = BatchNorm2d(depth[3],affine=False, track_running_stats=False)

        self.l_1 = Linear(in_features = depth[3], 
                          out_features = num_l1,
                          bias = False)
        
        self.maxpool = MaxPool2d(kernel_size = 2,
                                stride = 2)
    
    def forward(self, x): # x.size() = [batch, channel, height, width]
					      # after application becomes:
        x = relu(self.conv_1(x))              #[x,32,32,32]
        x = self.bn1(x)
        x = self.maxpool(x)                   #[x,32,16,16]
        x = relu(self.conv_2(x))              #[x,64,16,16]
        x = self.bn2(x)
        x = self.maxpool(x)                   #[x,64,8,8]
        x = relu(self.conv_3(x))              #[x,128,8,8]
        x = self.bn3(x)
        x = self.maxpool(x)		              #[x,128,4,4]
        x = relu(self.conv_4(x))              #[x,64,1,1]
        x = self.bn4(x)
        x = x.view(-1, depth[3])
        #x = relu(self.l_1(x))
        return softmax(self.l_1(x), dim=1)    #[x,10,1,1]

net = Net()
print(net)