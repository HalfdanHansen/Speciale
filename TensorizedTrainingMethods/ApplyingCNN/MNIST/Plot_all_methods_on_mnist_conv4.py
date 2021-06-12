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

l11 = ax1.plot(epoch, data.loc[0], 'r-')
l12 = ax1.plot(epoch, data.loc[2], 'b--')
l13 = ax1.plot(epoch, data.loc[4], 'm--')
l14 = ax1.plot(epoch, data.loc[5], 'g--')
l15 = ax1.plot(epoch, data.loc[1], 'b:')
l16 = ax1.plot(epoch, data.loc[3], 'm:')
l17 = ax1.plot(epoch, data.loc[6], 'g:')

l21 = ax2.plot(epoch, data.loc[7], 'r-')
l22 = ax2.plot(epoch, data.loc[9], 'b--')
l23 = ax2.plot(epoch, data.loc[11], 'm--')
l24 = ax2.plot(epoch, data.loc[12], 'g--')
l25 = ax2.plot(epoch, data.loc[8], 'b:')
l26 = ax2.plot(epoch, data.loc[10], 'm:')
l27 = ax2.plot(epoch, data.loc[13], 'g:')

ax1.set_xlabel('Epochs')
ax1.set_ylim([0.5,1])
ax1.set_ylabel('Training Accuracy')
ax2.set_ylabel('Validation  Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylim([0.5,1])
plt.suptitle('Results for conv4 with 50 epochs, MNIST', fontsize=20)

line_labels = ['BL', 'CD 3D' ,'BAF 3D' ,'ATCD 3D','CD 4D','BAF 4D', 'ATDC 4D']
fig.tight_layout(pad=2.0)
fig.legend([l11, l12, l13, l14, l15], 
           labels = line_labels,
           loc = 'center right',
           fontsize = 16,
           borderaxespad = 0.1)
plt.subplots_adjust(right=0.82)

plt.savefig('Plot_all_methods_on_mnist_conv4.pgf')
plt.show()

data = pd.read_csv(r'2905_MNIST100_conv4_4D2methods.csv',header=None,index_col=None)
# 0: 'Normal',1:'D4DD',2:'D3DD',3:'BAF4D',4:'BAF3D',5:'ATDC3D',6:'ATDC4D'

epoch = range(100)

fig1, (ax11, ax21) = plt.subplots(1, 2,figsize=(11,6))

l111 = ax11.plot(epoch, data.loc[0], 'r-')
l121 = ax11.plot(epoch, data.loc[1], 'b:')
l131 = ax11.plot(epoch, data.loc[2], 'm:')
l141 = ax11.plot(epoch, data.loc[3], 'g:')

l211 = ax21.plot(epoch, data.loc[0], 'r-')
l221 = ax21.plot(epoch, data.loc[1], 'b:')
l231 = ax21.plot(epoch, data.loc[2], 'm:')
l241 = ax21.plot(epoch, data.loc[3], 'g:')


ax11.set_xlabel('Epochs')
ax11.set_ylim([0,1])
ax11.set_ylabel('Training Accuracy')
ax21.set_ylabel('Validation  Accuracy')
ax21.set_xlabel('Epochs')
ax21.set_ylim([0,1])
plt.suptitle('Results for 4D methods on conv4 with 100 epochs, MNIST', fontsize=20)

line_labels = ['BL', 'CD 4D','BAF 4D', 'ATDC 4D']
fig1.tight_layout(pad=2.0)
fig1.legend([l111, l121, l131, l141], 
           labels = line_labels,
           loc = 'center right',
           fontsize = 16,
           borderaxespad = 0.1)
plt.subplots_adjust(right=0.82)

plt.savefig('Plot_4D_on_mnist_conv4.pgf')
plt.show()