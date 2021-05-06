import pandas as pd

#'Normal', 'D4DD' ,'D3DD' ,'BAF4D','BAF3D','ATDC3D','ATDC4D'

#data = pd.read_csv (r'/zhome/ab/6/109248/Desktop/Speciale/Speciale/TensorizedTrainingMethods/PackagesAndModels/MNIST_conv4_7methods.csv',header=None,index_col=None)
data = pd.read_csv (r'0505_MNIST_conv500_5methods.csv',header=None,index_col=None)

import matplotlib.pyplot as plt
epoch = range(50)

parameterFull = 3*1*3*3+6*3*3*3+12*6*3*3+3*12*3*3+7*7*3*10
parameter3D = 3*(1+3+3)+6*(3+3+3)+12*(6+3+3)+3*(12+3+3)+7*7*3*10
parameter4D = (1+3+3+3)+(3+3+3+6)+(6+3+3+12)+(12+3+3+3)+7*7*3*10

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(11,6))

ax1.plot(epoch, data.loc[0], 'r-',
         epoch, data.loc[2], 'b:',
         epoch, data.loc[1], 'b--',
         epoch, data.loc[3], 'm:',
         epoch, data.loc[4], 'm--')
         
ax2.plot(epoch, data.loc[5], 'r-',
         epoch, data.loc[7], 'b:',
         epoch, data.loc[6], 'b--',
         epoch, data.loc[8], 'm:',
         epoch, data.loc[9], 'm--')

ax1.legend(['Normal', 'D3DD', 'D4DD','ATDC3D','ATDC4D'], loc = 'lower right')
ax2.legend(['Normal', 'D3DD' ,'D4DD', 'ATDC3D', 'ATDC4D'], loc = 'lower right')
ax1.set_xlabel('Epochs')
ax1.set_ylim([0,1.1])
ax1.set_ylabel('Training Accuracy')
ax2.set_ylabel('Validation  Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylim([0,1.1])
plt.suptitle('Training Results For Methods on conv500 and MNIST', fontsize=14)

plt.savefig('Plot_all_methods_on_mnist_conv500.png')
plt.show()