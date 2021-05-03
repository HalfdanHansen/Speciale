#https://medium.com/jovianml/image-classification-with-cifar100-deep-learning-using-pytorch-9d9211a696e
if __name__ == '__main__':  
    import os
    import sys
    from copy import deepcopy
    from PackagesAndModels.pack import *
    from PackagesAndModels.method_functions import *
    from webNet_withoutres import *
    from PackagesAndModels.train_val_test_CIFAR10 import *

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
    epochs = 50
    
    #convName = ['conv_1','conv_2','conv_3','conv_4','conv_5','conv_6','conv_7','conv_8','conv_9','conv_10','conv_11']

    depthtrain = []
    depthtest = []
    depthloss = []
    
    depths = 10
    
    for i in range(depths):
        net = WebNet_noRes(conv_block, 3, 100, 1+i*3)
        net.to(device)
        
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
        
        depthtrain.append(train_acc)
        depthtest.append(test_acc)
        depthloss.append(losses)

    save_train = pd.DataFrame(depthtrain)
    save_test = pd.DataFrame(depthtest)
    save_loss = pd.DataFrame(depthloss)
    pd.concat([save_train,save_test,save_loss],axis = 0).to_csv('3004_webNet_deeper_littleres.csv',index=False,header=False)
