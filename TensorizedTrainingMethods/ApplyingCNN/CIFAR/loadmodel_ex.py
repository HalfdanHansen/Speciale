#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 12:56:37 2021

@author: s152576
"""
if __name__ == '__main__':    
    import os
    import sys
    from copy import deepcopy
    from PackagesAndModels.pack import *
    from PackagesAndModels.method_functions import *
    from PackagesAndModels.CIFAR_MODELS import *
    from PackagesAndModels.train_val_test_CIFAR10 import *
    import icecream

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
    
       # To load a trained model you need to define the network before, such that the system knows the format of this.
    net = torch.load("0705_conv500PARAFAC4DCIFAR10_rank1", map_location=torch.device("cpu"))
    net.cuda()
    net.eval()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=alpha)

    train_acc = []
    test_acc = []
    losses = []
    
    t = []
    for i in range(1000):
            start = time.time()
            evaluate_cifar(testloader, net).cpu().item()
            end = time.time()
            t.append(end-start)
    tmean = np.mean(t)
    
    
    print("Time done!")