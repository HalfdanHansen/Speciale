if __name__ == '__main__':    
    import os 
    print(os.getcwd())
    from pathlib import Path
    os.chdir(str(Path(os.getcwd()).parents[2]))
    os.chdir(os.getcwd()+'/PackagesAndModels')
    from train_val_test_CIFAR10 import *
    from method_functions import *
    from CIFAR_MODELS import *
    from pack import *

    trainloader, testloader = load_cifar()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    alpha = 0.01
    epochs = 50
    numModels = 5
    
    results_train = []
    results_test = []
    results_loss = []
    
    convName = ['conv_1','conv_2','conv_3','conv_4']
    
    net = convNet500
    net.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=alpha)
    
    for repeats in range(10):
        net.apply(weight_reset)

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
          
          if epoch > 1 and losses[-1]/losses[-2] > 0.9975:
            results_train.append(train_acc)
            results_test.append(test_acc)
            results_loss.append(losses)
            break
              
    save_train = pd.DataFrame(results_train)
    save_test = pd.DataFrame(results_test)
    save_loss = pd.DataFrame(results_loss)
    pd.concat([save_train,save_test,save_loss],axis = 0).to_csv('1604_CIFAR10_conv500.csv',index=False,header=False)
    

        
