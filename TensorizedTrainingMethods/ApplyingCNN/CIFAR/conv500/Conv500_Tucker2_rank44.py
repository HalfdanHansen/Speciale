import sys
import os
os.chdir('C:/Users/Halfdan/Desktop/specialemappe/GitHub/Speciale')

from train_val_test_CIFAR10 import *


if __name__ == '__main__':
    trainloader, testloader = load_cifar()
    alpha = 0.01

    criterion = nn.CrossEntropyLoss()

    epochs = 1

    net = convNet500
    net.cuda()
    convName = ['conv_1','conv_2','conv_3','conv_4']
    rank1 = 4
    rank2 = 4
    utc_convs = initialize_model_weights_from_Tucker2(convName,net,"net",rank1,rank2,[3,3,3,4])

    train_acc = []
    test_acc = []
    losses = []

    optimizer = optim.Adam(net.parameters(), lr=alpha)

    for epoch in range(epochs):
        running_loss = train_net_Tucker2(losses, net, "net",trainloader, criterion, optimizer, convName, utc_convs, alpha, rank1, rank2)

        net.eval()
        train_acc.append(evaluate_cifar(trainloader, net).cpu().item())
        test_acc.append(evaluate_cifar(testloader, net).cpu().item())
        losses.append(running_loss)
        
