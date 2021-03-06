if __name__ == '__main__':    
    import os 
    #print(os.getcwd())
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
    ranks = 5

    results_train = []
    results_test = []
    results_loss = []
    
    results_train = []
    results_test = []
    results_loss = []
    
    results_train = []
    results_test = []
    results_loss = []
    
    convName = ['conv_1','conv_2','conv_3','conv_4']
    lName = ['l_1']
    
    net = convNet500
    net.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=alpha)
    
    for rank in range(ranks):
        for repeats in range(10):
            net.apply(weight_reset)
    
            train_acc = []
            test_acc = []
            losses = []
            
            utc_convs = initialize_model_weights_from_Tucker2(convName,net,"net",rank+1,rank+1,[3,3,3,4])
  
            for epoch in range(epochs):
                running_loss = train_net_Tucker2_ATDC(losses, net, "net", trainloader, criterion, optimizer, convName, utc_convs, alpha, M+1, M+1,lName)
    
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
    pd.concat([save_train,save_test,save_loss],axis = 0).to_csv('1804_CIFAR10_Tucker24D_conv500_ATDC_rank.csv',index=False,header=False)
    

        
