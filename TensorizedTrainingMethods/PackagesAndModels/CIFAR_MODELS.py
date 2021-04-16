from pack import *

## PAPERNET ##

# hyperameters of the model
num_classes = 10
channels = 3
height = 32
width = 32

num_l1 = 10

num_filters_conv = [64,144,256]
kernel_size_conv = 3
padding_conv = 1
stride_conv = 1

class PaperNet(nn.Module):

    def __init__(self):
        super(PaperNet, self).__init__()
        
        self.conv_1 = Conv2d(in_channels = channels,
                            out_channels = num_filters_conv[0],
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)
        
        self.bn1 = BatchNorm2d(num_filters_conv[0])
        
        self.conv_2 = Conv2d(in_channels = num_filters_conv[0],
                            out_channels = num_filters_conv[0],
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)

        self.conv_3 = Conv2d(in_channels = num_filters_conv[0],
                            out_channels = num_filters_conv[1],
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)
        
        self.bn2 = BatchNorm2d(num_filters_conv[1])
        
        self.conv_4 = Conv2d(in_channels = num_filters_conv[1],
                            out_channels = num_filters_conv[1],
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)
        
        self.conv_5 = Conv2d(in_channels = num_filters_conv[1],
                            out_channels = num_filters_conv[2],
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)
        
        self.bn3 = BatchNorm2d(num_filters_conv[2])
        
        self.conv_6 = Conv2d(in_channels = num_filters_conv[2],
                            out_channels = num_filters_conv[2],
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)
        
        self.conv_7 = Conv2d(in_channels = num_filters_conv[2],
                            out_channels = 100,
                            kernel_size = kernel_size_conv,
                            stride = stride_conv,
                            padding = padding_conv,
                            bias = False)
                            
        self.bn4 = BatchNorm2d(100)
        
        self.l_1 = Linear(in_features = 100*4*4, 
                            out_features = num_l1,
                            bias = True)
        
        self.maxpool = MaxPool2d(kernel_size = 2,
                            stride = 2)
        
        self.dropout = Dropout2d(0.5)
    
    def forward(self, x):                     # x.size() = [batch, channel, height, width]
                    					      # after application becomes:
        x = relu(self.conv_1(x))              #[x,64,32,32]
        x = self.bn1(x)
        x = relu(self.conv_2(x))              #[x,64,32,32]
        x = self.maxpool(x)                   #[x,64,16,16]
        x = self.dropout(x)
        x = relu(self.conv_3(x))              #[x,144,16,16]
        x = self.bn2(x)
        x = relu(self.conv_4(x))              #[x,144,16,16]
        x = self.maxpool(x)                   #[x,144,8,8]
        x = self.dropout(x)
        x = relu(self.conv_5(x))              #[x,144,8,8]
        x = self.bn3(x)
        x = relu(self.conv_6(x))              #[x,256,8,8]
        x = self.maxpool(x)                   #[x,256,4,4]
        x = self.dropout(x)
        x = relu(self.conv_7(x))              #[x,256,1,1]
        x = self.bn4(x)
        x = x.view(-1, 100*4*4)
        return softmax(self.l_1(x), dim=1)    #[x,10,1,1]

papernet = PaperNet()

## CONVNET4

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

class ConvNet4(nn.Module):

    def __init__(self):
        super(ConvNet4, self).__init__()
        
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

convNet4 = ConvNet4()

## CONVNET4FC3 ##

#https://github.com/jamespengcheng/PyTorch-CNN-on-CIFAR10
# hyperameters of the model
num_classes = 10
channels = 3
height = 32
width = 32

num_l1 = 10

num_filters_conv = [16,32,64]
kernel_size_conv = 3
padding_conv = 1
stride_conv = 1

class ConvNet4FC3(nn.Module):

    def __init__(self):
        super(ConvNet4FC3, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3,3), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3,3), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3,3), padding=(1,1))
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(in_features=8*8*256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.Dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(in_features=64, out_features=10)
    
    def forward(self, x): # x.size() = [batch, channel, height, width]
					      # after application becomes:
        x = relu(self.conv1(x)) #32*32*48
        x = relu(self.conv2(x)) #32*32*96
        x = self.pool(x) #16*16*96
        x = self.Dropout(x)
        x = relu(self.conv3(x)) #16*16*192
        x = relu(self.conv4(x)) #16*16*256
        x = self.pool(x) # 8*8*256
        x = self.Dropout(x)
        x = x.view(-1, 8*8*256) # reshape x
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
        return x

convNet4fc3 = ConvNet4FC3()

## CONVNET500

num_classes = 10
channels = 3
height = 32
width = 32

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
                          bias = True)
        
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

convNet500 = ConvNet500()

## CONVNET5003D ##


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

class ConvNet500_3D(nn.Module):
    def __init__(self):
        super(ConvNet500_3D, self).__init__()

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

convNet500_3d = ConvNet500_3D()

## RESNET4 ##

class BasicBlock_ResNet4(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_ResNet4, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=False, track_running_stats=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes,affine=False, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes,affine=False, track_running_stats=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet4(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet4, self).__init__()
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

resNet4 = ResNet4(BasicBlock_ResNet4, [2, 2, 2, 2])

## CONVNET5004D

# hyperameters of the model


num_l1 = 10

depth = [32,64,128,64]
kernel_size_conv = 3
padding_conv = 1
stride_conv = 1

def conv_4D_block(in_f, out_f, kernelsize, stride, pad = 1, rank = 1):
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

class ConvNet500_4D(nn.Module):
    def __init__(self, rank = 1):
        super(ConvNet500_4D, self).__init__()

        self.conv_1_4D_block = conv_4D_block(3, depth[0], 3, 1, 1, rank = rank)
        self.bn1 = BatchNorm2d(depth[0], affine=False, track_running_stats=False)
        
        self.conv_2_4D_block = conv_4D_block(depth[0], depth[1], 3, 1, 1, rank = rank)
        self.bn2 = BatchNorm2d(depth[1], affine=False, track_running_stats=False)

        self.conv_3_4D_block = conv_4D_block(depth[1], depth[2], 3, 1, 1, rank = rank)
        self.bn3 = BatchNorm2d(depth[2], affine=False, track_running_stats=False)

        self.conv_4_4D_block = conv_4D_block(depth[2], depth[3], 4, 1, 0, rank = rank)
        self.bn4 = BatchNorm2d(depth[3], affine=False, track_running_stats=False)
        
        self.l_1 = Linear(in_features = depth[3], 
                          out_features = num_l1,
                          bias = True)

        self.maxpool = MaxPool2d(kernel_size = 2,
                                stride = 2)
    
    def forward(self, x): # x.size() = [batch, channel, height, width]
        x = relu(self.conv_1_4D_block(x))     #[x,32,32,32]
        x = self.bn1(x)
        x = self.maxpool(x)                   #[x,32,16,16]
        x = relu(self.conv_2_4D_block(x))     #[x,64,16,16]
        x = self.bn2(x)
        x = self.maxpool(x)                   #[x,64,8,8]
        x = relu(self.conv_3_4D_block(x))     #[x,128,8,8]
        x = self.bn3(x)
        x = self.maxpool(x)                   #[x,128,4,4]
        x = relu(self.conv_4_4D_block(x))     #[x,64,1,1]
        x = self.bn4(x)
        x = x.view(-1, depth[3])              #[x,64,1,1] -> [x,10,1,1]
        return softmax(self.l_1(x), dim=1)    #[x,10,1,1]

convNet500_4D = ConvNet500_4D(1)

## CONV500TUCKER2 ##

def conv_Tucker2_block(in_f, out_f, kernelsize, stride, pad = 1, rank1 = 1, rank2 = 1):
    return  nn.Sequential(
            nn.Conv2d(in_channels = in_f, out_channels = rank1, kernel_size=(1,1),
                      stride=(1,1), padding=0, bias=False),
        
            nn.Conv2d(in_channels = rank1, out_channels = rank2, kernel_size=(kernelsize,kernelsize),
                      stride=(stride,stride), padding=(pad,pad), bias=False),
        
            nn.Conv2d(in_channels = rank2, out_channels = out_f, kernel_size=(1,1),
                      stride=(1,1), padding=(0,0), bias=False)
        )

class ConvNet500_Tucker2(nn.Module):
    def __init__(self, rank1 = 1, rank2 = 1):
        super(ConvNet500_Tucker2, self).__init__()

        self.conv_1_Tucker2_block = conv_Tucker2_block(3, depth[0], 3, 1, 1, rank1 = rank1, rank2 = rank2)
        self.bn1 = BatchNorm2d(depth[0], affine=False, track_running_stats=False)
        
        self.conv_2_Tucker2_block = conv_Tucker2_block(depth[0], depth[1], 3, 1, 1, rank1 = rank1, rank2 = rank2)
        self.bn2 = BatchNorm2d(depth[1], affine=False, track_running_stats=False)

        self.conv_3_Tucker2_block = conv_Tucker2_block(depth[1], depth[2], 3, 1, 1, rank1 = rank1, rank2 = rank2)
        self.bn3 = BatchNorm2d(depth[2], affine=False, track_running_stats=False)

        self.conv_4_Tucker2_block = conv_Tucker2_block(depth[2], depth[3], 4, 1, 0, rank1 = rank1, rank2 = rank2)
        self.bn4 = BatchNorm2d(depth[3], affine=False, track_running_stats=False)
        
        self.l_1 = Linear(in_features = depth[3], 
                          out_features = num_l1,
                          bias = False)

        self.maxpool = MaxPool2d(kernel_size = 2,
                                stride = 2)
    
    def forward(self, x): # x.size() = [batch, channel, height, width]
        x = relu(self.conv_1_Tucker2_block(x))     #[x,32,32,32]
        x = self.bn1(x)
        x = self.maxpool(x)                        #[x,32,16,16]
        x = relu(self.conv_2_Tucker2_block(x))     #[x,64,16,16]
        x = self.bn2(x)
        x = self.maxpool(x)                        #[x,64,8,8]
        x = relu(self.conv_3_Tucker2_block(x))     #[x,128,8,8]
        x = self.bn3(x)
        x = self.maxpool(x)                        #[x,128,4,4]
        x = relu(self.conv_4_Tucker2_block(x))     #[x,64,1,1]
        x = self.bn4(x)
        x = x.view(-1, depth[3])                   #[x,64,1,1] -> [x,10,1,1]
        return softmax(self.l_1(x), dim=1)         #[x,10,1,1]

convNet500_Tucker211 = ConvNet500_Tucker2(1,1)
