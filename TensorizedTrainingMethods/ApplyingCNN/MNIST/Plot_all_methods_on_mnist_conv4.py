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
