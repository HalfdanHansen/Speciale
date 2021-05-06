#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:04:30 2021

@author: s152576
"""

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

    trainloader, testloader = load_cifar()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    alpha = 0.001
    epochs = 50
    
    convName = ['conv1[0]','conv2[0]','conv3[0]','conv4[0]','conv5[0]','conv6[0]','conv7[0]','conv8[0]','conv9[0]','conv10[0]','conv11[0]']
    lName = ["classifier[2]"]
    
    net = webNet_noResTucker211
    net.cuda()
    
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
    pd.concat([save_train,save_test,save_loss],axis = 0).to_csv('0305_webNet_noRes_Tucker24D_CIFAR10.csv',index=False,header=False)