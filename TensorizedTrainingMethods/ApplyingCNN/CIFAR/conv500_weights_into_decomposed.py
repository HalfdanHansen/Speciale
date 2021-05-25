from PackagesAndModels.pack import *
from PackagesAndModels.method_functions import *
from PackagesAndModels.train_val_test_CIFAR10 import *
from PackagesAndModels.CIFAR_MODELS import *
import pickle

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batchsize = 100

transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
# laoding the data
trainloader, testloader = load_cifar()

#%%
# ATCD 3D

#Load structure of decomposed network
net3D = deepcopy(convNet500_3d)
net3D.apply(weight_reset)

convNameDec = []
# getting the names of decomposed model
for name,i in net3D.named_parameters():
  print(name)
  name = name.replace(".0.","[0].")
  name = name.replace(".0.","[0].")
  name = name.replace(".1.","[1].")
  name = name.replace(".2.","[2].")
  name = "net3D." + name + ".data"
  convNameDec.append(name)

#Load trained full network with method
netATCD3D = deepcopy(convNet500)
netATCD3D = torch.load("0905_conv500ATCD3DCIFAR10", map_location=torch.device("cpu"))
netATCD3D.to(device)
netATCD3D.eval()

convName = []
# getting the names of full model
for name, layer in netATCD3D.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        convName.append(name)

# Decomposed weights of method network
pqt_convs = []
for c in convName:
  convData = eval("netATCD3D."+c+".weight.data")
  de_layer = []
  for cc in convData:
      de_layer.append(parafac(tl.tensor(cc), rank = 1)[1])
  pqt_convs.append(de_layer)
              
#Insert decompsed convolution weights into decomposed network structure
for l,layer in enumerate(pqt_convs):
  name1 = eval(convNameDec[l*3+0])
  name2 = eval(convNameDec[l*3+1])
  name3 = eval(convNameDec[l*3+2])

  temp1 = torch.zeros_like(name1)
  temp2 = torch.zeros_like(name2)
  temp3 = torch.zeros_like(name3)
  
  for f,fil in enumerate(layer):
    temp1[f] = torch.tensor(fil[0]).reshape(temp1.shape[1],temp1.shape[2],temp1.shape[3])
    temp2[f] = torch.tensor(fil[1]).reshape(temp2.shape[1],temp2.shape[2],temp2.shape[3])
    temp3[f] = torch.tensor(fil[2]).reshape(temp3.shape[1],temp3.shape[2],temp3.shape[3])
  
  name1[:] = temp1
  name2[:] = temp2
  name3[:] = temp3

#Instert linear weights into decompsed structure
net3D.l_1.weight.data[:] = netATCD3D.l_1.weight.data
net3D.l_1.bias.data[:] = netATCD3D.l_1.bias.data

#Test network
print("training accuracy for the ATDC3D model is " + str(evaluate_cifar(testloader, netATCD3D).cpu()))
print("training accuracy for the D3DD   model is " + str(evaluate_cifar(testloader, net3D).cpu()))

#%%

# ATCD 4D rank 1

#Load structure of decomposed network
net4D = deepcopy(convNet500_4D)
net4D.apply(weight_reset)

#Load trained full network with method
netATCD4D = deepcopy(convNet500)
netATCD4D = torch.load("0905_conv500ATCD4DCIFAR10", map_location=torch.device("cpu"))
netATCD4D.to(device)
netATCD4D.eval()

convNameDec = []
# getting the names of decomposed model
for name,i in net4D.named_parameters():
  name = name.replace(".0.","[0].")
  name = name.replace(".1.","[1].")
  name = name.replace(".2.","[2].")
  name = name.replace(".3.","[3].")
  name = "net4D." + name + ".data"
  convNameDec.append(name)

convName = []
# getting the names of full model
for name, layer in netATCD4D.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        convName.append(name)

# Decomposed weights of method network
pqtu_convs = []
for c in convName:
  convData = eval("netATCD4D."+c+".weight.data")
  pqtu_convs.append(parafac(tl.tensor(convData), rank = 1)[1])
              
#Insert decompsed convolution weights into decomposed network structure
for l,layer in enumerate(pqtu_convs):
  name1 = eval(convNameDec[l*4+0])
  name2 = eval(convNameDec[l*4+1])
  name3 = eval(convNameDec[l*4+2])
  name4 = eval(convNameDec[l*4+3])
  
  temp1 = torch.zeros_like(name1)
  temp2 = torch.zeros_like(name2)
  temp3 = torch.zeros_like(name3)
  temp4 = torch.zeros_like(name4)
  
  temp1 = torch.tensor(layer[1]).reshape(temp1.shape[0],temp1.shape[1],temp1.shape[2],temp1.shape[3])
  temp2 = torch.tensor(layer[2]).reshape(temp2.shape[0],temp2.shape[1],temp2.shape[2],temp2.shape[3])
  temp3 = torch.tensor(layer[3]).reshape(temp3.shape[0],temp3.shape[1],temp3.shape[2],temp3.shape[3])
  temp4 = torch.tensor(layer[0]).reshape(temp4.shape[0],temp4.shape[1],temp4.shape[2],temp4.shape[3])

  name1[:] = temp1
  name2[:] = temp2
  name3[:] = temp3
  name4[:] = temp4

#Instert linear weights into decompsed structure
net4D.l_1.weight.data[:] = netATCD4D.l_1.weight.data
net4D.l_1.bias.data[:] = netATCD4D.l_1.bias.data

#Test network
print("training accuracy for the ATDC4D model is " + str(evaluate_cifar(testloader, netATCD4D).cpu()))
print("training accuracy for the D4DD   model is " + str(evaluate_cifar(testloader, net4D).cpu()))

#%%

# ATCD 4D rank 8

#Load structure of decomposed network
net4D_r8 = deepcopy(convNet500_4D_rank8)
net4D_r8.apply(weight_reset)

convNameDec = []
# getting the names of decomposed model
for name,i in net4D_r8.named_parameters():
  name = name.replace(".0.","[0].")
  name = name.replace(".1.","[1].")
  name = name.replace(".2.","[2].")
  name = name.replace(".3.","[3].")
  name = "net4D_r8." + name + ".data"
  convNameDec.append(name)

#Load trained full network with method
netATCD4D_r8 = deepcopy(convNet500)
netATCD4D_r8 = torch.load("2005__conv500ATCD4DCIFAR10_rank8", map_location=torch.device("cpu"))
netATCD4D_r8.to(device)
netATCD4D_r8.eval()

convName = []
# getting the names of full model
for name, layer in netATCD4D_r8.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        convName.append(name)

# Decomposed weights of method network
#pqtu_convs = []
#for c in convName:
#  convData = eval("netATCD4D_r8."+c+".weight.data")
#  pqtu_convs.append(parafac(tl.tensor(convData), rank = 8)[1])
 
pqtu_convs = pickle.load(open("2005_conv500ATCD4DCIFAR10_pqtu_rank8.p", "rb"))

#Insert decompsed convolution weights into decomposed network structure
for l,layer in enumerate(pqtu_convs):
  name1 = eval(convNameDec[l*4+0])
  name2 = eval(convNameDec[l*4+1])
  name3 = eval(convNameDec[l*4+2])
  name4 = eval(convNameDec[l*4+3])
  
  temp1 = torch.zeros_like(name1)
  temp2 = torch.zeros_like(name2)
  temp3 = torch.zeros_like(name3)
  temp4 = torch.zeros_like(name4)
  
  temp1 = torch.tensor(layer[1]).reshape(temp1.shape[0],temp1.shape[1],temp1.shape[2],temp1.shape[3])
  temp2 = torch.tensor(layer[2]).reshape(temp2.shape[0],temp2.shape[1],temp2.shape[2],temp2.shape[3])
  temp3 = torch.tensor(layer[3]).reshape(temp3.shape[0],temp3.shape[1],temp3.shape[2],temp3.shape[3])
  temp4 = torch.tensor(layer[0]).reshape(temp4.shape[0],temp4.shape[1],temp4.shape[2],temp4.shape[3])
  
  name1[:] = temp1
  name2[:] = temp2
  name3[:] = temp3
  name4[:] = temp4

#Instert linear weights into decompsed structure
net4D_r8.l_1.weight.data[:] = netATCD4D_r8.l_1.weight.data
net4D_r8.l_1.bias.data[:] = netATCD4D_r8.l_1.bias.data

#Test network
print("training accuracy for the ATDC4D rank 8 model is " + str(evaluate_cifar(testloader, netATCD4D_r8).cpu()))
print("training accuracy for the D4DD  rank 8  model is " + str(evaluate_cifar(testloader, net4D_r8).cpu()))

#%%

# Tucker2 ATCD 4D rank 1

#Load structure of decomposed network
netTucker2 = deepcopy(convNet500_Tucker211)
netTucker2.apply(weight_reset)

#Load trained full network with method
netTucker2ATCD = deepcopy(convNet500)
netTucker2ATCD = torch.load("0905_conv500Tucker2ATCD4DCIFAR10_rank11", map_location=torch.device("cpu"))
netTucker2ATCD.to(device)
netTucker2ATCD.eval()

convNameDec = []
# getting the names of decomposed model
for name,i in netTucker2.named_parameters():
  name = name.replace(".0.","[0].")
  name = name.replace(".1.","[1].")
  name = name.replace(".2.","[2].")
  name = "netTucker2." + name + ".data"
  convNameDec.append(name)

convName = []
# getting the names of full model
for name, layer in netTucker2ATCD.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        convName.append(name)

# Decomposed weights of method network
utc_convs = []
for c in convName:
  convData = eval("netTucker2ATCD."+c+".weight.data")
  core, de_layer = tucker(tl.tensor(convData), rank = [1,1])[1]
  utc_convs.append(de_layer)
              
#Insert decompsed convolution weights into decomposed network structure
for l,layer in enumerate(utc_convs):
  name1 = eval(convNameDec[l*3+0])
  name2 = eval(convNameDec[l*3+1])
  name3 = eval(convNameDec[l*3+2])
  
  temp1 = torch.zeros_like(name1)
  temp2 = torch.zeros_like(name2)
  temp3 = torch.zeros_like(name3)
  
  for f,fil in enumerate(layer):
      temp1 = torch.tensor(fil[1]).reshape(temp1.shape[0],temp1.shape[1],temp1.shape[2],temp1.shape[3])
      temp2 = torch.tensor(fil[2]).reshape(temp2.shape[0],temp2.shape[1],temp2.shape[2],temp2.shape[3])
      temp3 = torch.tensor(fil[0]).reshape(temp3.shape[0],temp3.shape[1],temp3.shape[2],temp3.shape[3])
      
  name1[:] = temp1
  name2[:] = temp2
  name3[:] = temp3

#Instert linear weights into decompsed structure
netTucker2.l_1.weight.data[:] = netTucker2ATCD.l_1.weight.data
netTucker2.l_1.bias.data[:] = netTucker2ATCD.l_1.bias.data

#Test network
print("training accuracy for the Tucker2ATDC model is " + str(evaluate_cifar(testloader, netATCD4D).cpu()))
print("training accuracy for the Tucker2   model is " + str(evaluate_cifar(testloader, net4D).cpu()))

