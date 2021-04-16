if __name__ == '__main__':
    import os 
    print(os.getcwd())
    import sys 
    
    #import re
    
    #from pathlib import Path
    #os.chdir(str(Path(os.getcwd()).parents[2]))
    #os.chdir(os.getcwd()+'/PackagesAndModels')
    #print(os.getcwd())
    from PackagesAndModels.pack import *
    
    #import importlib
    #fol = re.sub("/", ".", d)[1:-1]
    #importlib.import_module(fol+".pack")
    
    from method_functions import *
    from CIFAR_MODELS import *
    from train_val_test_CIFAR10 import *
    
    trainloader, testloader = load_cifar()
    
    alpha = 0.001
    
    criterion = nn.CrossEntropyLoss()
    
    results_train = []
    results_test = []
    results_loss = []
    epochs = 50
    
    NumModels = 5
    
    for M in range(NumModels):
        for repeats in range(10):
            net = ConvNet500_4D(M+1)
            net.cuda()
    
            train_acc = []
            test_acc = []
            losses = []
    
            optimizer = optim.Adam(net.parameters(), lr=alpha)
            for epoch in range(epochs):
              running_loss = 0
              net.train()
    
              for i, data in enumerate(trainloader, 0):
    
                inputs, labels = data[0].to(device), data[1].to(device)
    
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
    
              net.eval()
              train_acc.append(evaluate_cifar(trainloader, net).cpu().item())
              test_acc.append(evaluate_cifar(testloader, net).cpu().item())
              losses.append(running_loss)
              
              if epoch > 1 and losses[-1]/losses[-2] > 0.975:
                results_train.append(train_acc)
                results_test.append(test_acc)
                results_loss.append(losses)
                break
                
    save_train = pd.DataFrame(results_train)
    save_test = pd.DataFrame(results_test)
    save_loss = pd.DataFrame(results_loss)
    pd.concat([save_train,save_test,save_loss],axis = 0).to_csv('1504_CIFAR10_D4DD_PARAFAC_conv500_rank.csv',index=False,header=False)
    
    
    
