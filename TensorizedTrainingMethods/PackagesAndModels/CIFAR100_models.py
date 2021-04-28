from .pack import *

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

def normal_convlayer(in_f, out_f, kernelsize = 3, stride = 1, pad = 1, rank = 1, regfunc = None):
    layer = [nn.Conv2d(in_channels = in_f,
               out_channels = out_f,
               kernel_size = kernel_size_conv,
               stride = stride_conv,
               padding = padding_conv,
               bias = False),
               nn.BatchNorm2d(out_f), 
               nn.ReLU(inplace=True)]
    if regfunc == "m":
        layer.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
    elif regfunc == "d1":
        layer.append(nn.Dropout2d(0.4))
    elif regfunc == "d2":
        layer.append(nn.Dropout2d(0.5))
    return nn.Sequential(*layer)


def conv_3D_block(in_f, out_f, kernelsize = 3, stride = 1, pad = 1, rank = 1, regfunc = "m"):
    layer = [nn.Sequential(
            nn.Conv2d(in_channels = in_f, out_channels = out_f, kernel_size=1,
                      stride=1, padding=0, bias=False, groups = 1),
            nn.Conv2d(in_channels = out_f, out_channels = out_f, kernel_size=(kernelsize,1),
                      stride=1, padding=(1,0), bias=False, groups = out_f),
            nn.Conv2d(in_channels = out_f, out_channels = out_f, kernel_size=(1,kernelsize),
                      stride=1, padding=(0,1), bias=False, groups = out_f)
            ),
               nn.BatchNorm2d(out_f), 
               nn.ReLU(inplace=True)]
    if regfunc == "m":
        layer.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
    elif regfunc == "d1":
        layer.append(nn.Dropout2d(0.4))
    elif regfunc == "d2":
        layer.append(nn.Dropout2d(0.5))
    return nn.Sequential(*layer) 

def conv_4D_block(in_f, out_f, kernelsize = 3, stride = 1, pad = 1, rank = 1, regfunc = "m"):
    layer = [nn.Sequential(
          nn.Conv2d(in_channels = in_f, out_channels = rank, kernel_size=(1,1),
                    stride=(1,1), padding=0, bias=False),
      
          nn.Conv2d(in_channels = rank, out_channels = rank, kernel_size=(kernelsize,1),
                    stride=(stride,1), padding=(pad,0), bias=False, groups = rank),
      
          nn.Conv2d(in_channels = rank, out_channels = rank, kernel_size=(1,kernelsize),
                    stride=(1,stride), padding=(0,pad), bias=False, groups = rank),
      
          nn.Conv2d(in_channels = rank, out_channels = out_f, kernel_size=(1,1),
                    stride=(1,1), padding=(0,0), bias=False)
          ),
         nn.BatchNorm2d(out_f), 
         nn.ReLU(inplace=True)]
    if regfunc == "m":
        layer.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
    elif regfunc == "d1":
        layer.append(nn.Dropout2d(0.4))
    elif regfunc == "d2":
        layer.append(nn.Dropout2d(0.5))
    return nn.Sequential(*layer) 



def conv_Tucker2_block(in_f, out_f, kernelsize = 3, stride = 1, pad = 1, rank = [1,1], regfunc = "m"):
    layer = [nn.Sequential(
            nn.Conv2d(in_channels = in_f, out_channels = rank[0], kernel_size=(1,1),
                      stride=(1,1), padding=0, bias=False),
        
            nn.Conv2d(in_channels = rank[0], out_channels = rank[1], kernel_size=(kernelsize,kernelsize),
                      stride=(stride,stride), padding=(pad,pad), bias=False),
        
            nn.Conv2d(in_channels = rank[1], out_channels = out_f, kernel_size=(1,1),
                      stride=(1,1), padding=(0,0), bias=False)
        ),
         nn.BatchNorm2d(out_f), 
         nn.ReLU(inplace=True)]
    if regfunc == "m":
        layer.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
    elif regfunc == "d1":
        layer.append(nn.Dropout2d(0.4))
    elif regfunc == "d2":
        layer.append(nn.Dropout2d(0.5))
    return nn.Sequential(*layer) 



class PaperNet4(nn.Module):

    def __init__(self, func, kernelsize, stride, pad, rank):
        super().__init__()
        
        self.conv_1 = func(channels,num_filters_conv[0], regfunc = "d2")
        self.conv_2 = func(num_filters_conv[0], num_filters_conv[1], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank, regfunc = "m")
        self.conv_3 = func(num_filters_conv[1], num_filters_conv[2], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank, regfunc = "d1")
        self.conv_4 = func(num_filters_conv[2], num_filters_conv[3], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank, regfunc = "m")
        self.conv_5 = func(num_filters_conv[3], num_filters_conv[4], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank, regfunc = "d1")
        self.conv_6 = func(num_filters_conv[4], num_filters_conv[5], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank, regfunc = "d1")
        self.conv_7 = func(num_filters_conv[5], num_filters_conv[6], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank, regfunc = "m")
        self.conv_8 = func(num_filters_conv[6], num_filters_conv[7], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank, regfunc = "d1")
        self.conv_9 = func(num_filters_conv[7], num_filters_conv[8], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank, regfunc = "d1")
        self.conv_10 = func(num_filters_conv[8], num_filters_conv[9], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank, regfunc = "m")
        self.conv_11 = func(num_filters_conv[9], num_filters_conv[10], kernelsize=kernelsize, stride=stride, pad=pad, rank=rank, regfunc = "d1")

        self.bnl_1 = BatchNorm1d(num_perceptrons_fc[1])
        self.bnl_2 = BatchNorm1d(num_perceptrons_fc[2])
        self.bnl_3 = BatchNorm1d(num_perceptrons_fc[3])

        self.l_1 = Linear(in_features = num_perceptrons_fc[0], out_features = num_perceptrons_fc[1], bias = True)
        self.l_2 = Linear(in_features = num_perceptrons_fc[1], out_features = num_perceptrons_fc[2], bias = True)
        self.l_3 = Linear(in_features = num_perceptrons_fc[2], out_features = num_perceptrons_fc[3], bias = True)
        self.l_4 = Linear(in_features = num_perceptrons_fc[3], out_features = num_perceptrons_fc[4], bias = True)
        
        self.dropout = Dropout(0.5)
    
    def forward(self, x):                     # x.size() = [batch, channel, height, width]
                    					      # after application becomes:
        x = self.conv_1(x)              #[x,64,32,32]
        x = self.conv_2(x)              #[x,64,32,32]
        x = self.conv_3(x)              #[x,144,16,16]
        x = self.conv_4(x)              #[x,144,16,16]
        x = self.conv_5(x)              #[x,256,8,8]
        x = self.conv_6(x)              #[x,256,8,8]
        x = self.conv_7(x)         #[x,256,8,8]
        x = self.conv_8(x)              #[x,484,4,4]
        x = self.conv_9(x)            #[x,484,4,4]
        x = self.conv_10(x)             #[x,484,4,4]
        x = self.conv_11(x)              #[x,484,2,2]

        x = x.view(-1, num_perceptrons_fc[0]) #[x,484*4,1,1] #nn.Flatten()
        x = self.l_1(x)               #[x,2048,1,1]
        x = self.bnl_1(x)
        x = relu(x)
        x = self.dropout(x)
        x = self.l_2(x)               #[x,1024,1,1]
        x = self.bnl_2(x)
        x = relu(x)
        x = self.dropout(x)
        x = self.l_3(x)              #[x,512,1,1]
        x = self.bnl_3(x)
        x = relu(x)  
        x = self.dropout(x)
        return softmax(self.l_4(x), dim=1)    #[x,100,1,1]

paperNet4 = PaperNet4(normal_convlayer, 3, 1, 1, 1)

#paperNet43D = PaperNet4(conv_3D_block, 3, 1, 1, 1)

#paperNet44D = PaperNet4(conv_4D_block, 3, 1, 1, 1)

#paperNet4Tucker2 = PaperNet4(conv_Tucker2_block, 3, 1, 1, [1,1])
