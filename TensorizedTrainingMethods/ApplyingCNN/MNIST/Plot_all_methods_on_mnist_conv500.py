import pandas as pd

#0: 'Normal', 1: 'D4DD' ,2: 'D3DD', 3:'ATDC3D', 4:'ATDC4D'

#data = pd.read_csv (r'/zhome/ab/6/109248/Desktop/Speciale/Speciale/TensorizedTrainingMethods/PackagesAndModels/MNIST_conv4_7methods.csv',header=None,index_col=None)
data = pd.read_csv(r'0905_MNIST_conv500_5methods.csv',header=None,index_col=None)

import matplotlib.pyplot as plt
epoch = range(50)

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(11,6))

ax1.plot(epoch, data.loc[0], 'r-',
         epoch, data.loc[2], 'b--',
         epoch, data.loc[3], 'g--',
         epoch, data.loc[1], 'b:',
         epoch, data.loc[4], 'g:')
         
ax2.plot(epoch, data.loc[5], 'r-',
         epoch, data.loc[7], 'b--',
         epoch, data.loc[8], 'g--',
         epoch, data.loc[6], 'b:',
         epoch, data.loc[9], 'g:')

ax2.legend(['Normal', 'CD 3D', 'ATDC 3D', 'CD 4D' ,'ATDC 4D'], loc = 'center right')
ax1.set_xlabel('Epochs')
ax1.set_ylim([0,1.1])
ax1.set_ylabel('Training Accuracy')
ax2.set_ylabel('Validation  Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylim([0,1.1])
plt.suptitle('Training Results For Methods on conv500 and MNIST', fontsize=14)
plt.savefig('1105_Plot_all_methods_on_mnist_conv500.png')
plt.show()