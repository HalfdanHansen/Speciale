import os 
import sys 
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'0905_MNIST_conv4_7methods.csv',header=None,index_col=None)
# 0: 'Normal',1:'D4DD',2:'D3DD',3:'BAF4D',4:'BAF3D',5:'ATDC3D',6:'ATDC4D'

epoch = range(50)

parameterFull = 3*1*3*3+6*3*3*3+12*6*3*3+3*12*3*3+7*7*3*10
parameter3D = 3*(1+3+3)+6*(3+3+3)+12*(6+3+3)+3*(12+3+3)+7*7*3*10
parameter4D = (1+3+3+3)+(3+3+3+6)+(6+3+3+12)+(12+3+3+3)+7*7*3*10

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(11,6))

ax1.plot(epoch, data.loc[0], 'r-',
         epoch, data.loc[2], 'b--',
         epoch, data.loc[4], 'm--',
         epoch, data.loc[5], 'g--',
         epoch, data.loc[1], 'b:',
         epoch, data.loc[3], 'm:',
         epoch, data.loc[6], 'g:')

ax2.plot(epoch, data.loc[7], 'r-',
         epoch, data.loc[9], 'b--',
         epoch, data.loc[11], 'm--',
         epoch, data.loc[12], 'g--',
         epoch, data.loc[8], 'b:',
         epoch, data.loc[10], 'm:',
         epoch, data.loc[13], 'g:')

ax1.legend(['Normal', 'CD 3D' ,'BAF 3D' ,'ATCD 3D','CD 4D','BAF 4D', 'ATDC 4D'], loc = 'lower right')
ax1.set_xlabel('Epochs')
ax1.set_ylim([0.5,1])
ax1.set_ylabel('Training Accuracy')
ax2.set_ylabel('Validation  Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylim([0.5,1])
plt.suptitle('Training Results For Methods on conv4 and MNIST', fontsize=14)

plt.savefig('1105_Plot_all_methods_on_mnist_conv4.png')
plt.show()


#Test time
import time
import os 
print(os.getcwd())
import sys 
from pathlib import Path
os.chdir(str(Path(os.getcwd()).parents[1]))
os.chdir(os.getcwd()+'\PackagesAndModels')
from pack import *
from method_functions import *
from MNIST_MODELS import *
from train_val_test_MNIST import *

os.chdir(str(Path(os.getcwd()).parents[0]))
print(os.getcwd())
os.chdir(os.getcwd()+'\ApplyingCNN\MNIST')

time_list = []

netATCD3D = torch.load("0905_conv4ATDC3DMNIST", map_location=torch.device("cpu"))
netATCD3D.eval()
t = []
for i in range(1000):
    start = time.time()
    preds, test_acc_full = evaluate_test(x_test, targets_test, netATCD3D)
    end = time.time()
    t.append(end-start)
tmean = np.mean(t)
time_list.append(tmean)

netATCD4D = torch.load("0905_conv4ATDC4DMNIST", map_location=torch.device("cpu"))
netATCD4D.eval()
t = []
for i in range(1000):
    start = time.time()
    preds, test_acc_full = evaluate_test(x_test, targets_test, netATCD4D)
    end = time.time()
    t.append(end-start)
tmean = np.mean(t)
time_list.append(tmean)
netnormal = torch.load("0905_conv4normalMNIST", map_location=torch.device("cpu"))
netnormal.eval()
t = []
for i in range(1000):
    start = time.time()
    preds, test_acc_full = evaluate_test(x_test, targets_test, netnormal)
    end = time.time()
    t.append(end-start)
tmean = np.mean(t)
time_list.append(tmean)
netBAF3D = torch.load("0905_conv4BAF3DMNIST", map_location=torch.device("cpu"))
netBAF3D.eval()
t = []
for i in range(1000):
    start = time.time()
    preds, test_acc_full = evaluate_test(x_test, targets_test, netBAF3D)
    end = time.time()
    t.append(end-start)
tmean = np.mean(t)
time_list.append(tmean)
netBAF4D = torch.load("0905_conv4BAF3DMNIST", map_location=torch.device("cpu"))
netBAF4D.eval()
t = []
for i in range(1000):
    start = time.time()
    preds, test_acc_full = evaluate_test(x_test, targets_test, netBAF4D)
    end = time.time()
    t.append(end-start)
tmean = np.mean(t)
time_list.append(tmean)
netD3DD = torch.load("0905_conv43DMNIST", map_location=torch.device("cpu"))
netD3DD.eval()
t = []
for i in range(1000):
    start = time.time()
    preds, test_acc_full = evaluate_test(x_test, targets_test, netD3DD)
    end = time.time()
    t.append(end-start)
tmean = np.mean(t)
time_list.append(tmean)
netD4DD = torch.load("0905_conv4D4DDMNIST", map_location=torch.device("cpu"))
netD4DD.eval()
t = []
for i in range(1000):
    start = time.time()
    preds, test_acc_full = evaluate_test(x_test, targets_test, netD4DD)
    end = time.time()
    t.append(end-start)
tmean = np.mean(t)
time_list.append(tmean)
