if __name__ == '__main__':

    import os
    import sys
    from copy import deepcopy
    
    from PackagesAndModels.pack import *
    from PackagesAndModels.method_functions import *
    from Papernet2model import *
    from PackagesAndModels.train_val_test_CIFAR10 import *

    trainloader, testloader = load_cifar()

    batchsize = 100
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    alpha = 0.001
    epochs = 50
    
    # choose network block type to run
    net = deepcopy(papernet2True3DSharedWeights)#
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

    save_train = pd.DataFrame(train_acc)
    save_test = pd.DataFrame(test_acc)
    save_loss = pd.DataFrame(losses)
    pd.concat([save_train,save_test,save_loss],axis = 0).to_csv('07_05_Papernet2True3DSharedWeights_CIFAR10.csv',index=False,header=False)
    
    torch.save(net,"Papernet2True3DSharedWeights")

    
    