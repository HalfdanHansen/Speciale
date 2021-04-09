from pack import *
from MNIST_subset import *

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
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

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
        # torch.Tensor.view: http://pytorch.org/docs/master/tensors.html?highlight=view#torch.Tensor.view
        #   Returns a new tensor with the same data as the self tensor,
        #   but of a different size.
        # the size -1 is inferred from other dimensions 
        
        x = x.view(-1, self.l1_in_features)
        #x = self.dropout(relu(self.l_1(x)))
        x = relu(self.l_1(x))
        return softmax(self.l_out(x), dim=1)


net = Net()
print(net)