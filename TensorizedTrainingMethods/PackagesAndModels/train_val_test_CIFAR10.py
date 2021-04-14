from pack import *
from method_functions import *

def evaluate_cifar(loader,model):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  correct = 0
  total = 0
  with torch.no_grad():
    for data in loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().detach() #item()
  return (correct / total)

def train_net_Tucker2(losses, net, netname,trainloader, criterion, optimizer, convName, utc_convs, alpha):
  running_loss = 0
  net.train()

  for i, data in enumerate(trainloader, 0):
    inputs, labels = data[0].to(device), data[1].to(device)
    
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    #optimizer.step()

    #ATDC step for convolutional layers
    for k1,utc in enumerate(utc_convs):

      convGrad = eval(netname+"."+convName[k1]+".weight.grad")
      convData = eval(netname+"."+convName[k1]+".weight.data")

      utc_convs[k1] = ATDC_update_step_Tucker2(
                      ATDC_get_grads_Tucker2(convGrad, utc), alpha, utc)
      convData[:] = torch.einsum('hq,sw,wqij->hsij',utc[0],utc[1],utc[2])

    #normal step for linear layer
    net.l_1.weight.data[:] = torch.sub(net.l_1.weight.data,net.l_1.weight.grad, alpha = alpha)
    net.l_1.bias.data[:] = torch.sub(net.l_1.bias.data,net.l_1.bias.grad,alpha = alpha)

    running_loss += loss.item()

  return running_loss

def load_cifar():
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
  return (trainloader,testloader)
