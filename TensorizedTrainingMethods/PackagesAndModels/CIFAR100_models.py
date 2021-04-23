from PackagesAndModels.pack import *

## PAPERNET ##

# parameters of the model
num_classes = 100
channels = 3

num_filters_conv = [64,64,144,144,256,256,256,484,484,484,484]
num_perceptrons_fc = [484*4,2048,1024,512,100]
kernel_size_conv = 3
padding_conv = 1
stride_conv = 1

# Defining the different types of layers

def normal_convlayer(in_f, out_f, kernelsize = 3, stride = 1, pad = 1, rank = 1):
    return Conv2d(  in_channels = in_f,
                    out_channels = out_f,
                    kernel_size = kernel_size_conv,
                    stride = stride_conv,
                    padding = padding_conv,
                    bias = False)

def conv_3D_block(in_f, out_f, kernelsize = 3, stride = 1, pad = 1, rank = 1):
    return  nn.Sequential(
            nn.Conv2d(in_channels = in_f, out_channels = out_f, kernel_size=1,
                      stride=1, padding=0, bias=False, groups = 1),
            nn.Conv2d(in_channels = out_f, out_channels = out_f, kernel_size=(kernelsize,1),
                      stride=1, padding=(1,0), bias=False, groups = out_f),
            nn.Conv2d(in_channels = out_f, out_channels = out_f, kernel_size=(1,kernelsize),
                      stride=1, padding=(0,1), bias=False, groups = out_f)
            )

def conv_4D_block(in_f, out_f, kernelsize = 3, stride = 1, pad = 1, rank = 1):
    return  nn.Sequential(
            nn.Conv2d(in_channels = in_f, out_channels = rank, kernel_size=(1,1),
                      stride=(1,1), padding=0, bias=False),
        
            nn.Conv2d(in_channels = rank, out_channels = rank, kernel_size=(kernelsize,1),
                      stride=(stride,1), padding=(pad,0), bias=False, groups = rank),
        
            nn.Conv2d(in_channels = rank, out_channels = rank, kernel_size=(1,kernelsize),
                      stride=(1,stride), padding=(0,pad), bias=False, groups = rank),
        
            nn.Conv2d(in_channels = rank, out_channels = out_f, kernel_size=(1,1),
                      stride=(1,1), padding=(0,0), bias=False)
        )

def conv_Tucker2_block(in_f, out_f, kernelsize = 3, stride = 1, pad = 1, rank = [1,1]):
    return  nn.Sequential(
            nn.Conv2d(in_channels = in_f, out_channels = rank[0], kernel_size=(1,1),
                      stride=(1,1), padding=0, bias=False),
        
            nn.Conv2d(in_channels = rank[0], out_channels = rank[1], kernel_size=(kernelsize,kernelsize),
                      stride=(stride,stride), padding=(pad,pad), bias=False),
        
            nn.Conv2d(in_channels = rank[1], out_channels = out_f, kernel_size=(1,1),
                      stride=(1,1), padding=(0,0), bias=False)
        )

class PaperNet4(nn.Module):

    def __init__(self, func, kernelsize, stride, pad, rank):
        super(PaperNet4, self).__init__()
        
        self.conv_1 = func(channels,num_filters_conv[0])
        self.conv_2 = func(num_filters_conv[0], num_filters_conv[1], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank)
        self.conv_3 = func(num_filters_conv[1], num_filters_conv[2], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank)
        self.conv_4 = func(num_filters_conv[2], num_filters_conv[3], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank)
        self.conv_5 = func(num_filters_conv[3], num_filters_conv[4], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank)
        self.conv_6 = func(num_filters_conv[4], num_filters_conv[5], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank)
        self.conv_7 = func(num_filters_conv[5], num_filters_conv[6], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank)
        self.conv_8 = func(num_filters_conv[6], num_filters_conv[7], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank)
        self.conv_9 = func(num_filters_conv[7], num_filters_conv[8], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank)
        self.conv_10 = func(num_filters_conv[8], num_filters_conv[9], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank)
        self.conv_11 = func(num_filters_conv[9], num_filters_conv[10], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank)

        self.bn1 = BatchNorm2d(num_filters_conv[0],affine=False, track_running_stats=False)
        self.bn2 = BatchNorm2d(num_filters_conv[1],affine=False, track_running_stats=False)
        self.bn3 = BatchNorm2d(num_filters_conv[2],affine=False, track_running_stats=False)
        self.bn4 = BatchNorm2d(num_filters_conv[3],affine=False, track_running_stats=False)
        self.bn5 = BatchNorm2d(num_filters_conv[4],affine=False, track_running_stats=False)
        self.bn6 = BatchNorm2d(num_filters_conv[5],affine=False, track_running_stats=False)
        self.bn7 = BatchNorm2d(num_filters_conv[6],affine=False, track_running_stats=False)
        self.bn8 = BatchNorm2d(num_filters_conv[7],affine=False, track_running_stats=False)
        self.bn9 = BatchNorm2d(num_filters_conv[8],affine=False, track_running_stats=False)
        self.bn10 = BatchNorm2d(num_filters_conv[9],affine=False, track_running_stats=False)
        self.bn11 = BatchNorm2d(num_filters_conv[10],affine=False, track_running_stats=False)

        self.l_1 = Linear(in_features = num_perceptrons_fc[0], out_features = num_perceptrons_fc[1], bias = True)
        self.l_2 = Linear(in_features = num_perceptrons_fc[1], out_features = num_perceptrons_fc[2], bias = True)
        self.l_3 = Linear(in_features = num_perceptrons_fc[2], out_features = num_perceptrons_fc[3], bias = True)
        self.l_4 = Linear(in_features = num_perceptrons_fc[3], out_features = num_perceptrons_fc[4], bias = True)
        
        self.maxpool = MaxPool2d(kernel_size = 2, stride = 2)
        
        self.dropout1 = Dropout2d(0.4)
        self.dropout2 = Dropout2d(0.5)
    
    def forward(self, x):                     # x.size() = [batch, channel, height, width]
                    					      # after application becomes:
        x = relu(self.conv_1(x))              #[x,64,32,32]
        x = self.bn1(x)
        x = self.dropout2(x)

        x = relu(self.conv_2(x))              #[x,64,32,32]
        x = self.bn2(x)
        x = self.maxpool(x)                   #[x,64,16,16]
        
        x = relu(self.conv_3(x))              #[x,144,16,16]
        x = self.bn3(x)
        x = self.dropout1(x)

        x = relu(self.conv_4(x))              #[x,144,16,16]
        x = self.bn4(x)
        x = self.maxpool(x)                   #[x,144,8,8]

        x = relu(self.conv_5(x))              #[x,256,8,8]
        x = self.bn5(x)
        x = self.dropout1(x)

        x = relu(self.conv_6(x))              #[x,256,8,8]
        x = self.bn6(x)
        x = self.dropout1(x)

        x = relu(self.conv_7(x))              #[x,256,8,8]
        x = self.bn7(x)
        x = self.maxpool(x)                   #[x,256,4,4]

        x = relu(self.conv_8(x))              #[x,484,4,4]
        x = self.bn8(x)
        x = self.dropout1(x)

        x = relu(self.conv_9(x))              #[x,484,4,4]
        x = self.bn9(x)
        x = self.dropout1(x)

        x = relu(self.conv_10(x))              #[x,484,4,4]
        x = self.bn10(x)
        x = self.maxpool(x)                   #[x,484,2,2]

        x = relu(self.conv_11(x))              #[x,484,2,2]
        x = self.bn11(x)
        x = self.dropout1(x)

        x = x.view(-1, num_perceptrons_fc[0]) #[x,484*4,1,1]
        x = relu(self.l_1(x))               #[x,2048,1,1]
        x = self.dropout2(x)
        x = relu(self.l_2(x))               #[x,1024,1,1]
        x = self.dropout2(x)
        x = relu(self.l_3(x))               #[x,512,1,1]
        x = self.dropout2(x)
        return softmax(self.l_4(x), dim=1)    #[x,100,1,1]

paperNet4 = PaperNet4(normal_convlayer, 3, 1, 1, 1)

paperNet43D = PaperNet4(conv_3D_block, 3, 1, 1, 1)

paperNet44D = PaperNet4(conv_4D_block, 3, 1, 1, 1)

paperNet4Tucker2 = PaperNet4(conv_Tucker2_block, 3, 1, 1, [1,1])
