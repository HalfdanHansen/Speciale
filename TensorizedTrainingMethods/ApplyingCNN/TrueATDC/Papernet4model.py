from PackagesAndModels.pack import *

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, regul):
        super(conv_block, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias = False)
        self.batch = nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        
        if regul == 'pool':     self.after = nn.MaxPool2d(2)
        elif regul == 'drop1':  self.after = nn.Dropout2d(0.4)
        elif regul == 'drop2':  self.after = nn.Dropout2d(0.5)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)
        x = self.after(x)
        return x

class conv_3D_block(nn.Module):
    def __init__(self, in_c, out_c, regul):
        super(conv_3D_block, self).__init__()
        
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=False, groups = 1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=(3,1), padding=(1,0), bias=False, groups = out_c)
        self.conv3 = nn.Conv2d(out_c, out_c, kernel_size=(1,3), padding=(0,1), bias=False, groups = out_c)
        
        self.batch = nn.BatchNorm2d(out_c, affine=False, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        
        if regul == 'pool':     self.after = nn.MaxPool2d(2)
        elif regul == 'drop1':  self.after = nn.Dropout2d(0.4)
        elif regul == 'drop2':  self.after = nn.Dropout2d(0.5)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.batch(x)
        x = self.relu(x)
        x = self.after(x)
        return x

class conv_3D_block_shared_weights(nn.Module):
    def __init__(self, in_channels, out_channels, regul):
        super(conv_3D_block_shared_weights, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=1,
                          padding=0, bias=False, groups = 1)
        '''self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size=(3,1),
                          padding=(1,0), bias=False, groups = out_channels)
        self.conv3 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size=(1,3),
                          padding=(0,1), bias=False, groups = out_channels)
        
        #del self.conv2.weight
        #del self.conv3.weight'''
        
        self.root = int(np.sqrt(out_channels))
        self.OUT = out_channels
        
        self.p = nn.Parameter(torch.randn(self.root,3), requires_grad=True)
        self.q = nn.Parameter(torch.randn(self.root,3), requires_grad=True)
        
        self.batch = nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        
        if regul == 'pool':     self.after = nn.MaxPool2d(2)
        elif regul == 'drop1':  self.after = nn.Dropout2d(0.4)
        elif regul == 'drop2':  self.after = nn.Dropout2d(0.5)
        
    def forward(self, x):
        x = self.conv1(x)
        
        '''numlist = torch.linspace(1,self.root,self.root)
        onelist = torch.ones(self.root)
        onenums = torch.einsum('i,j -> ij', numlist, onelist)
        index1 = onenums.flatten().byte()
        index2 = torch.transpose(onenums,0,1).flatten().byte()'''
        
        a = torch.ones([self.OUT,1,3,1]).cuda()
        b = torch.ones([self.OUT,1,1,3]).cuda()
        
        for i in range(self.root):
            for j in range(self.root):
                a[i+j,0,:,0] = self.p[i,:]
                b[i+j,0,0,:] = self.q[i,:]

        
        x = torch.nn.functional.conv2d(x, a, bias = None, padding = (1,0), groups = self.OUT)
        x = torch.nn.functional.conv2d(x, b, bias = None, padding = (0,1), groups = self.OUT)
        print(type(x))
        
        x = self.batch(x)
        x = self.relu(x)
        x = self.after(x)
        return x

class Papernet4True(nn.Module):
    def __init__(self, block, classes):
            super(Papernet4True, self).__init__()
            
            self.conv1 = self._make_layer(block, 3, 64, 'drop2')
            self.conv2 = self._make_layer(block, 64, 64, 'pool')
            self.conv3 = self._make_layer(block, 64, 144, 'drop1')
            self.conv4 = self._make_layer(block, 144, 144, 'pool')
            self.conv5 = self._make_layer(block, 144, 256, 'drop1')
            self.conv6 = self._make_layer(block, 256, 256, 'drop1')
            self.conv7 = self._make_layer(block, 256, 256, 'pool')
            self.conv8 = self._make_layer(block, 256, 484, 'drop1')
            self.conv9 = self._make_layer(block, 484, 484, 'drop1')
            self.conv10 = self._make_layer(block, 484, 484, 'pool')
            self.conv11 = self._make_layer(block, 484, 484, 'drop1')

            self.classifier = nn.Sequential(nn.Flatten(),
                                            nn.Linear(4*484, 2048),
                                            nn.BatchNorm1d(2048, affine=False, track_running_stats=False),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(2048, 1024),
                                            nn.BatchNorm1d(1024, affine=False, track_running_stats=False),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(1024, 512),
                                            nn.BatchNorm1d(512, affine=False, track_running_stats=False),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(512, classes)
                                            #nn.Softmax()
                                            )
            
    def _make_layer(self, block, in_channels, out_channels, regul):
        layers = []
        layers.append(block(in_channels, out_channels, regul))
        return nn.Sequential(*layers)
    
    def forward(self, xb):
            out = self.conv1(xb)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.conv5(out)
            out = self.conv6(out)
            out = self.conv7(out)
            out = self.conv8(out)
            out = self.conv9(out)
            out = self.conv10(out)
            out = self.conv11(out)
            out = self.classifier(out)
            
            return out


papernet4True = Papernet4True(conv_block, 100)
papernet4True3D = Papernet4True(conv_3D_block, 100)
papernet4True3DSharedWeights = Papernet4True(conv_3D_block_shared_weights, 100)
