import sys
sys.path.insert(1,'TensorizedTrainingMethods/PackagesAndModels')

from train_val_test_CIFAR10 import *


trainloader, testloader = load_cifar()
alpha = 0.01

criterion = nn.CrossEntropyLoss()

epochs = 1

net = convNet500
net.cuda()
convName = ['conv_1','conv_2','conv_3','conv_4']

utc_convs = initialize_model_weights_from_Tucker2(convName,net,"net",2,2,[3,3,3,4])

train_acc = []
test_acc = []
losses = []

optimizer = optim.Adam(net.parameters(), lr=alpha)

for epoch in range(epochs):
    running_loss = train_net_Tucker2(losses, net, "net",trainloader, criterion, optimizer, convName, utc_convs, alpha)

    net.eval()
    train_acc.append(evaluate_cifar(trainloader, net).cpu().item())
    test_acc.append(evaluate_cifar(testloader, net).cpu().item())
    losses.append(running_loss)
    
