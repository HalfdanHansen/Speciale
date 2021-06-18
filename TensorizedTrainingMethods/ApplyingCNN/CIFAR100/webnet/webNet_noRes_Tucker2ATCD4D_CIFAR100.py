if __name__ == '__main__':    
    import os
    import sys
    from copy import deepcopy
    from PackagesAndModels.pack import *
    from PackagesAndModels.method_functions import *
    from webNet_withoutres import *
    from PackagesAndModels.train_val_test_CIFAR10 import *
    import icecream

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
    
    alpha = 0.1
    epochs = 50
    
    convName = ['conv1[0]','conv2[0]','conv3[0]','conv4[0]','conv5[0]','conv6[0]','conv7[0]','conv8[0]','conv9[0]','conv10[0]','conv11[0]']
    lName = ["classifier[2]"]
    
    net = webNet_noRes
    net.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=alpha)

    train_acc = []
    test_acc = []
    losses = []

    #pqtu_convs = initialize_model_weights_from_PARAFAC_rank(convName, net, "net", 10)
    
    utc_convs = initialize_model_weights_from_Tucker2(convName, net, "net", 1, 1, [3,3,3,4])

    for epoch in range(epochs):
        #losses, net, netname,trainloader, criterion, optimizer, convName, utc_convs, alpha, rank1, rank2, lName
        running_loss = train_net_Tucker2_ATDC(losses, net, "net", trainloader, criterion, optimizer, convName, utc_convs, alpha, 1, 1, lName)

        net.eval()
        train_acc.append(evaluate_cifar(trainloader, net).cpu().item())
        test_acc.append(evaluate_cifar(testloader, net).cpu().item())
        losses.append(running_loss)
    
    save_train = pd.DataFrame(train_acc)
    save_test = pd.DataFrame(test_acc)
    save_loss = pd.DataFrame(losses)
    pd.concat([save_train,save_test,save_loss],axis = 0).to_csv('0905_webNetTucker2ATCD4DCIFAR100_rank11.csv',index=False,header=False)
    
     #Save model
    torch.save(net, "0905_webNetTucker2ATCD4DCIFAR100_rank11")