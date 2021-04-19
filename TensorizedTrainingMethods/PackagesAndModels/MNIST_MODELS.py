from pack import *
from train_val_test_MNIST import *

#Conv 2 network

# hyperameters of the model
num_classes = 10
channels = x_train.shape[1]
height = x_train.shape[2]
width = x_train.shape[3]

num_filters_conv1 = 16
kernel_size_conv1 = 5 # [height, width]
stride_conv = 1 # [stride_height, stride_width]
num_l1 = 100
padding_conv1 = 1

num_filters_conv2 = 8
kernel_size_conv2 = 5 # [height, width]
stride_conv2 = 1 # [stride_height, stride_width]
padding_conv2 = 1

def compute_conv_dim(dim_size, kernel_size_conv, padding_conv, stride_conv):
    return int((dim_size - kernel_size_conv + 2 * padding_conv) / stride_conv + 1) 

# define network
class ConvNet2(nn.Module):

    def __init__(self):
        super(ConvNet2, self).__init__()

        # out_dim = (input_dim - filter_dim + 2padding) / stride + 1
        
        self.conv_1 = Conv2d(in_channels=channels,
                            out_channels=num_filters_conv1,
                            kernel_size=kernel_size_conv1,
                            stride=stride_conv,
                            padding = padding_conv1,
                            bias = False)

        self.bn1 = BatchNorm2d(num_filters_conv1)
        
        self.conv1_out_height = compute_conv_dim(height, kernel_size_conv1, padding_conv1, stride_conv)
        self.conv1_out_width = compute_conv_dim(width, kernel_size_conv1, padding_conv1, stride_conv)

        self.conv_2 = Conv2d(in_channels=num_filters_conv1,
                            out_channels=num_filters_conv2,
                            kernel_size=kernel_size_conv2,
                            stride=stride_conv2,
                            padding = padding_conv2,
                            bias = False)
        
        self.bn2 = BatchNorm2d(num_filters_conv2)
        
        self.conv2_out_height = compute_conv_dim(self.conv1_out_height, kernel_size_conv2, padding_conv2, stride_conv2)
        self.conv2_out_width = compute_conv_dim(self.conv1_out_width, kernel_size_conv2, padding_conv2, stride_conv2)


        self.l1_in_features = num_filters_conv2 * self.conv2_out_height * self.conv2_out_width
        
        self.l_1 = Linear(in_features=self.l1_in_features, 
                          out_features=num_l1,
                          bias=True)
        
        self.l_out = Linear(in_features=num_l1, 
                            out_features=num_classes,
                            bias=False)
    
    def forward(self, x): # x.size() = [batch, channel, height, width]
        x = relu(self.conv_1(x))
        x = self.bn1(x)
        x = relu(self.conv_2(x))
        x = self.bn2(x)
        x = x.view(-1, self.l1_in_features)
        x = relu(self.l_1(x))
        return softmax(self.l_out(x), dim=1)


convNet2 = ConvNet2()

#Conv500 network
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

class ConvNet500(nn.Module):

    def __init__(self):
        super(ConvNet500, self).__init__()
        
        self.conv_1 = Conv2d(in_channels = channels,
                            out_channels = depth[0],
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)

        self.bn1 = BatchNorm2d(depth[0],affine=False,track_running_stats=False)
        
        self.conv_2 = Conv2d(in_channels = depth[0],
                            out_channels = depth[1],
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)
        
        self.bn2 = BatchNorm2d(depth[1],affine=False,track_running_stats=False)

        self.conv_3 = Conv2d(in_channels = depth[1],
                            out_channels = depth[2],
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)
        
        self.bn3 = BatchNorm2d(depth[2],affine=False,track_running_stats=False)

        self.conv_4 = Conv2d(in_channels = depth[2],
                            out_channels = depth[3],
                            kernel_size = 7,
                            stride = 1,
                            padding = 0,
                            bias = False)
        
        self.bn4 = BatchNorm2d(depth[3],affine=False,track_running_stats=False)
        
        self.l_1 = Linear(in_features = depth[3], 
                          out_features = num_l1,
                          bias = False)

        self.maxpool = MaxPool2d(kernel_size = 2,
                                stride = 2)
    
    def forward(self, x): # x.size() = [batch, channel, height, width]
        x = relu(self.conv_1(x))              #[x,32,28,28]
        x = self.bn1(x)
        x = self.maxpool(x)                   #[x,32,14,14]
        x = relu(self.conv_2(x))              #[x,64,14,14]
        x = self.bn2(x)
        x = self.maxpool(x)                   #[x,64,7,7]
        x = relu(self.conv_3(x))              #[x,128,7,7]
        x = self.bn3(x)
        x = relu(self.conv_4(x))              #[x,64,1,1]
        x = self.bn4(x)
        x = x.view(-1, depth[3])   #[x,64,1,1] -> [x,10,1,1]
        return softmax(self.l_1(x), dim=1)    #[x,10,1,1]

convNet500 = ConvNet500()

#ConvNet5003D

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

class ConvNet5003D(nn.Module):
    def __init__(self):
        super(ConvNet5003D, self).__init__()

        self.conv_1_3D_block = conv_3D_block(1,depth[0],3,depth[0])
        
        self.bn1 = BatchNorm2d(depth[0],affine=False,track_running_stats=False)
        
        self.conv_2_3D_block = conv_3D_block(depth[0],depth[1],3,depth[1])
        
        self.bn2 = BatchNorm2d(depth[1],affine=False,track_running_stats=False)

        self.conv_3_3D_block = conv_3D_block(depth[1],depth[2],3,depth[2])
        
        self.bn3 = BatchNorm2d(depth[2],affine=False,track_running_stats=False)

        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels = depth[2], out_channels = depth[3], kernel_size=1,
                      stride=1, padding=0, bias=False, groups = 1),
            nn.Conv2d(in_channels = depth[3], out_channels = depth[3], kernel_size=(7,1),
                      stride=1, padding=0, bias=False, groups = depth[3]),
            nn.Conv2d(in_channels = depth[3], out_channels = depth[3], kernel_size=(1,7),
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
        x = relu(self.conv_4(x))              #[x,64,4,4]
        x = self.bn4(x)
        x = x.view(-1, depth[3])   #[x,64,1,1] -> [x,10,1,1]
        return softmax(self.l_1(x), dim=1)    #[x,10,1,1]

convNet5003D = ConvNet5003D()

# Conv net 4

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

class ConvNet4(nn.Module):

    def __init__(self):
        super(ConvNet4, self).__init__()
        
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

convNet4 = ConvNet4()

# conv net 4 3d

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

class ConvNet43D(nn.Module):
    def __init__(self):
        super(ConvNet43D, self).__init__()

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

convNet43D = ConvNet43D()

#ConvNet5004D

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

def conv_4D_block(in_f, out_f, kernelsize, stride, pad = 1):
    return  nn.Sequential(
            nn.Conv2d(in_channels = in_f, out_channels = 1, kernel_size=1,
                      stride=(1,1), padding=0, bias=False),
            nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size=(kernelsize,1),
                      stride=(stride,1), padding=(pad,0), bias=False),
            nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size=(1,kernelsize),
                      stride=(1,stride), padding=(0,pad), bias=False),
            nn.Conv2d(in_channels = 1, out_channels = out_f, kernel_size=(1,1),
                      stride=(1,1), padding=(0,0), bias=False)
        )

class ConvNet5004D(nn.Module):
    def __init__(self):
        super(ConvNet5004D, self).__init__()

        self.conv_1_4D_block = conv_4D_block(1,depth[0], 3, 1, 1)
        
        self.bn1 = BatchNorm2d(depth[0],affine=False,track_running_stats=False)
        
        self.conv_2_4D_block = conv_4D_block(depth[0],depth[1], 3, 1, 1)
        
        self.bn2 = BatchNorm2d(depth[1],affine=False,track_running_stats=False)

        self.conv_3_4D_block = conv_4D_block(depth[1],depth[2], 3, 1, 1)
        
        self.bn3 = BatchNorm2d(depth[2],affine=False,track_running_stats=False)

        self.conv_4_4D_block = conv_4D_block(depth[2],depth[3], 7, 1, 0)

        self.bn4 = BatchNorm2d(depth[3],affine=False,track_running_stats=False)
        
        self.l_1 = Linear(in_features = depth[3], 
                          out_features = num_l1,
                          bias = False)

        self.maxpool = MaxPool2d(kernel_size = 2,
                                stride = 2)
    
    def forward(self, x): # x.size() = [batch, channel, height, width]
        x = relu(self.conv_1_4D_block(x))     #[x,32,28,28]
        x = self.bn1(x)
        x = self.maxpool(x)                   #[x,32,14,14]
        x = relu(self.conv_2_4D_block(x))     #[x,64,14,14]
        x = self.bn2(x)
        x = self.maxpool(x)                   #[x,64,7,7]
        x = relu(self.conv_3_4D_block(x))     #[x,128,7,7]
        x = self.bn3(x)
        x = relu(self.conv_4_4D_block(x))     #[x,64,1,1]
        x = self.bn4(x)
        x = x.view(-1, depth[3])   #[x,64,1,1] -> [x,10,1,1]
        return softmax(self.l_1(x), dim=1)    #[x,10,1,1]

convNet5004D = ConvNet5004D()


# conv net 4 4d

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

def conv_4D_block(in_f, out_f, kernelsize, stride, pad = 1):
    return  nn.Sequential(
            nn.Conv2d(in_channels = in_f, out_channels = 1, kernel_size=1,
                      stride=(1,1), padding=0, bias=False),
            nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size=(kernelsize,1),
                      stride=(stride,1), padding=(pad,0), bias=False),
            nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size=(1,kernelsize),
                      stride=(1,stride), padding=(0,pad), bias=False),
            nn.Conv2d(in_channels = 1, out_channels = out_f, kernel_size=(1,1),
                      stride=(1,1), padding=(0,0), bias=False)
        )

class ConvNet44D(nn.Module):
    def __init__(self):
        super(ConvNet44D, self).__init__()

        self.conv_1_4D_block = conv_4D_block(1,3,3,3)
        
        self.bn1 = BatchNorm2d(3)
        
        self.conv_2_4D_block = conv_4D_block(3,6,3,6)
        
        self.bn2 = BatchNorm2d(6)

        self.conv_3_4D_block = conv_4D_block(6,12,3,12)
        
        self.bn3 = BatchNorm2d(12)

        self.conv_4_4D_block = conv_4D_block(12,3,3,3)
        
        self.bn4 = BatchNorm2d(3)
        
        self.l_1 = Linear(in_features = 7*7*3, 
                          out_features = num_l1,
                          bias = False)

        self.maxpool = MaxPool2d(kernel_size = 2,
                                stride = 2)
    
    def forward(self, x): # x.size() = [batch, channel, height, width]
        x = relu(self.conv_1_4D_block(x))     #[x,3,28,28]
        x = self.bn1(x)
        x = self.maxpool(x)                   #[x,3,14,14]
        x = relu(self.conv_2_4D_block(x))     #[x,6,14,14]
        x = self.bn2(x)
        x = self.maxpool(x)                   #[x,6,7,7]
        x = relu(self.conv_3_4D_block(x))     #[x,12,7,7]
        x = self.bn3(x)
        x = relu(self.conv_4_4D_block(x))     #[x,3,7,7]
        x = self.bn4(x)
        x = x.view(-1, 7*7*3)   #[x,3*7*7,1,1] -> [x,10,1,1]
        return softmax(self.l_1(x), dim=1)    #[x,10,1,1]

convNet44D = ConvNet44D()