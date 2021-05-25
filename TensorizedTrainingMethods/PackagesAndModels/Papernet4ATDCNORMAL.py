if __name__ == '__main__':

    import os
    import sys
    from copy import deepcopy
    
    from Papernet4model import *

    from PackagesAndModels.pack import *
    from PackagesAndModels.train_val_test_CIFAR10 import *
    import time

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    batchsize = 100

    trainloader, testloader = load_cifar()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    alpha = 0.1
    epochs = 50
    
    convName = ['conv1[0].conv',
                 'conv2[0].conv',
                 'conv3[0].conv',
                 'conv4[0].conv',
                 'conv5[0].conv',
                 'conv6[0].conv',
                 'conv7[0].conv',
                 'conv8[0].conv',
                 'conv9[0].conv',
                 'conv10[0].conv',
                 'conv11[0].conv']
    
    lName = ['classifier[1]', 'classifier[5]', 'classifier[9]', 'classifier[13]']
    
    net = deepcopy(papernet4True)
    net.to(device)
    
    netname = "net"
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=alpha)

    train_acc = []
    test_acc = []
    losses = []

    pqt_convs = initialize_model_weights_from_PARAFAC3D_rank(convName, net, "net")

    for epoch in range(epochs):
        running_loss = train_net_PARAFAC3D_ATDC(losses, net, "net", trainloader, criterion, optimizer, convName, pqt_convs, alpha, 8, lName)

        net.eval()
        train_acc.append(evaluate_cifar(trainloader, net).cpu().item())        
        test_acc.append(evaluate_cifar(testloader, net).cpu().item())
        losses.append(running_loss)
        
    t = []
    for i in range(1000):
        start = time.time()
        evaluate_cifar(testloader, net).cpu().item()
        end = time.time()
        t.append(end-start)
    tmean = np.mean(t)

    
    save_train = pd.DataFrame(train_acc)
    save_test = pd.DataFrame(test_acc)
    save_loss = pd.DataFrame(losses)
    print("Time:" + str(tmean))
    pd.concat([save_train,save_test,save_loss],axis = 0).to_csv('14_05_Papernet4True3DATCD.csv',index=False,header=False)
    
    torch.save(net,"Papernet4True3DATCD")
