if __name__ == '__main__':    
    import os 
    print(os.getcwd())
    from pathlib import Path
    #os.chdir(str(Path(os.getcwd()).parents[2]))
    #os.chdir(os.getcwd()+'/PackagesAndModels')
    from PackagesAndModels.train_val_test_CIFAR10 import *
    from PackagesAndModels.method_functions import *
    from PackagesAndModels.CIFAR_MODELS import *
    from PackagesAndModels.pack import *

    trainloader, testloader = load_cifar()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    alpha = 0.01
    epochs = 50
    numModels = 5
    
    convName = ['conv_1','conv_2','conv_3','conv_4']
    
    net = convNet500
    net.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=alpha)
    
    NumModels = 5
    
    for M in range(NumModels):
        for repeats in range(10):
            net.apply(weight_reset)
    
            train_acc = []
            test_acc = []
            losses = []
            
            pqtu_convs = initialize_model_weights_from_PARAFAC_rank(convName,net,"net",M+1)
  
            for epoch in range(epochs):
                running_loss = train_net_PARAFAC4D_ATDC(losses, net, "net", trainloader, criterion, optimizer, convName, pqtu_convs, alpha, rank)
        
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
    pd.concat([save_train,save_test,save_loss],axis = 0).to_csv('1604_CIFAR10_PARAFAC4D_conv500_ATDC_rank.csv',index=False,header=False)
    

        
