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
                                                shuffle=True, num_workers = 2, pin_memory=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                            shuffle=False, num_workers =  2, pin_memory=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
<<<<<<< HEAD
    alpha = 0.1
=======
    alpha = 0.01
>>>>>>> 23da4ca8f1479e2159c22c29b19c4904bec3433e
    epochs = 5
    
    convName = ['conv1[0]','conv2[0]','conv3[0]','conv4[0]','conv5[0]','conv6[0]','conv7[0]','conv8[0]','conv9[0]','conv10[0]','conv11[0]']
    lName = ["classifier[2]"]
    
    net = deepcopy(webNet_noRes)
    net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=alpha)

    train_acc = []
    test_acc = []
    losses = []

    pqt_convs = initialize_model_weights_from_PARAFAC3D_rank(convName, net, "net")

    for epoch in range(epochs):
        running_loss = train_net_PARAFAC3D_ATDC(losses, net, "net", trainloader, criterion, optimizer, convName, pqt_convs, alpha, 1, lName)

        net.eval()
        train_acc.append(evaluate_cifar(trainloader, net).cpu().item())
        test_acc.append(evaluate_cifar(testloader, net).cpu().item())
        losses.append(running_loss)

    save_train = pd.DataFrame(train_acc)
    save_test = pd.DataFrame(test_acc)
    save_loss = pd.DataFrame(losses)
    pd.concat([save_train,save_test,save_loss],axis = 0).to_csv('2904_webNet_noRes_ATDC3D_CIFAR100_uniforminit.csv',index=False,header=False)
