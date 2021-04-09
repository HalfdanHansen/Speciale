#CNN CIFAR-10
#https://github.com/jamespengcheng/PyTorch-CNN-on-CIFAR10
from pack import *

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

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
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

net = Net()
print(net)