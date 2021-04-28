   
if __name__ == '__main__':    
    import os
    import sys
    from copy import deepcopy
    from PackagesAndModels.pack import *
    from PackagesAndModels.method_functions import *
    from PackagesAndModels.train_val_test_CIFAR10 import *
        
    def normal_conv(in_f, out_f):
        return nn.Sequential(nn.Conv2d(in_channels = in_f, out_channels = out_f,
                             kernel_size = 3, stride = 1, padding = 1, bias = False),
                             nn.BatchNorm2d(out_f),
                             nn.ReLU(inplace=True))
    
    def normal_block(in_f, out_f):
        return nn.Sequential(normal_conv(in_f, out_f),
                             normal_conv(out_f,out_f),
                             nn.MaxPool2d(kernel_size = 2, stride = 2))
    
    class kdnugnet(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.conv_1 = normal_block(3, 64)
            self.conv_2 = normal_block(64, 128)
            self.conv_3 = normal_block(128, 256)
            self.conv_4 = normal_block(256, 512)
            self.conv_5 = normal_block(512, 1024)
    
            self.l_1 = Linear(in_features = 1024, out_features = 100, bias = True)
            
            self.drop = Dropout2d(0.5)
        
        def forward(self, x):
            x = self.conv_1(x)
            x = self.conv_2(x)
            x = self.conv_3(x)
            x = self.conv_4(x)
            x = self.conv_5(x)
            x = x.view(-1, 1024)
            x = self.l_1(x)
            return softmax(x, dim=1)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    batchsize = 100

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                                shuffle=True, num_workers = 2, pin_memory=True) #numworker =2 and not pin_memory

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                            shuffle=False, num_workers =  2, pin_memory=True) #numworker =2 and not pin_memory
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    alpha = 0.001
    epochs = 5
    
    convName = ['conv_1','conv_2','conv_3','conv_4','conv_5','conv_6','conv_7','conv_8','conv_9','conv_10','conv_11']
    
    net = kdnugnet().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=alpha)

    train_acc = []
    test_acc = []
    losses = []

    for epoch in range(epochs):
      running_loss = 0
      net.train()

      for i, data in enumerate(trainloader, 0):

        inputs, labels = data[0].to(device), data[1].to(device)
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

      net.eval()
      train_acc.append(evaluate_cifar(trainloader, net).cpu().item())
      test_acc.append(evaluate_cifar(testloader, net).cpu().item())
      losses.append(running_loss)

    save_train = pd.DataFrame(train_acc)
    save_test = pd.DataFrame(test_acc)
    save_loss = pd.DataFrame(losses)
    pd.concat([save_train,save_test,save_loss],axis = 0).to_csv('2604_CIFAR100_KDnuggetNet.csv',index=False,header=False)

