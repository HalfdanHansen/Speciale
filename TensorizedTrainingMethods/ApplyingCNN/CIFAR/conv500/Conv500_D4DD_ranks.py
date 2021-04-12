from sklearn.metrics import accuracy_score
from copy import deepcopy

import sys
sys.path.insert(1,'TensorizedTrainingMethods/PackagesAndModels')

from pack import *
from CIFAR_MODELS import *
from train_val_test_CIFAR10 import *
from method_functions import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

batchsize = 100

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                          shuffle=True, num_workers = 2, pin_memory=True) #numworker =2 and not pin_memory

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                         shuffle=False, num_workers =  2, pin_memory=True) #numworker =2 and not pin_memory

alpha = 0.001

criterion = nn.CrossEntropyLoss()

results_train = []
results_test = []
results_loss = []
epochs = 1

NumModels = 6

for M in range(NumModels):
    for repeats in range(5):
        if M < 3:
            net = ConvNet500_4D((M+1)*2)
        else:
            net = ConvNet500_Tucker2((M+1)*2,(M+1)*2)

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
pd.concat([save_train,save_test,save_loss],axis = 0).to_csv('0704_CIFAR10_no_pass_net_D4DD_rank123.csv',index=False,header=False)