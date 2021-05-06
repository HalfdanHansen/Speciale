if __name__ == '__main__':

    import os
    import sys
    from copy import deepcopy
    
    from PackagesAndModels.pack import *
    from PackagesAndModels.method_functions import *
    from PackagesAndModels.CIFAR_MODELS import *
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
    
    convName = ['conv_1','conv_2','conv_3','conv_4']
    lName = ['l_1']
    
    net = convNet500
    net.cuda()
    
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
        test_acc.append(evaluate_cifar(testloader, net).cpu().item())
        end = time.time()
        t.append(end-start)
    tmean = np.mean(t)

    save_train = pd.DataFrame(train_acc)
    save_test = pd.DataFrame(test_acc)
    save_loss = pd.DataFrame(losses)
    print("Time rank 8:" + str(tmean))
    pd.concat([save_train,save_test,save_loss],axis = 0).to_csv('0505_conv500ATCD3DCIFAR10_rank8.csv',index=False,header=False)
