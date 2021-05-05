#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:58:02 2021

@author: s152576
"""

import pandas as pd

#Normal
#Rank 1: PARAFAC3D, ATCD3D, PARAFAC4D, Tucker24D, ATCD4D, Tucker2ATCD4D
#Rank 8:  PARAFAC4D, Tucker24D, ATCD4D, Tucker2ATCD4D

normal = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0305_conv500normalCIFAR10.csv',header=None,index_col=None)
PARAFAC3D_R1 = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0305_conv500PARAFAC3DCIFAR10.csv',header=None,index_col=None)
ATCD3D_R1 = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0305_conv500ATCD3DCIFAR10.csv',header=None,index_col=None)
PARAFAC4D_R1 = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0305_conv500PARAFAC4DCIFAR10.csv',header=None,index_col=None)
PARAFAC4D_R8 = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0305_conv500PARAFAC4DCIFAR10_rank8.csv',header=None,index_col=None)
Tucker24D_R1 = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0305_conv500Tucker2114DCIFAR10.csv',header=None,index_col=None)
Tucker24D_R8 = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0305_conv500Tucker2884DCIFAR10.csv',header=None,index_col=None)
ATCD4D_R1 = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0305_conv500ATCD4DCIFAR10.csv',header=None,index_col=None)
ATCD4D_R8 = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0305_conv500ATCD4DCIFAR10_rank8.csv',header=None,index_col=None)


import matplotlib.pyplot as plt
epoch = range(50)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(11,6))

ax1.plot(epoch, normal[0:50], 'r-',
         epoch, PARAFAC3D_R1[0:50], 'y--',
         epoch, ATCD3D_R1[0:50], 'b--',
         epoch, PARAFAC4D_R1[0:50], 'm:',
         epoch, PARAFAC4D_R8[0:50], 'm--',
         epoch, Tucker24D_R1[0:50], 'g--',
         epoch, Tucker24D_R8[0:50], 'g:',
         epoch, ATCD4D_R1[0:50], 'o--',
         epoch, ATCD4D_R8[0:50], 'o:')

ax2.plot(epoch, normal[51:100], 'r-',
         epoch, PARAFAC3D_R1[51:100], 'y--',
         epoch, ATCD3D_R1[51:100], 'b--',
         epoch, PARAFAC4D_R1[51:100], 'm:',
         epoch, PARAFAC4D_R8[51:100], 'm--',
         epoch, Tucker24D_R1[51:100], 'g--',
         epoch, Tucker24D_R8[51:100], 'g:',
         epoch, ATCD4D_R1[51:100], 'o--',
         epoch, ATCD4D_R8[51:100], 'o:')

ax2.plot(epoch, normal[101:150], 'r-',
         epoch, PARAFAC3D_R1[101:150], 'y--',
         epoch, ATCD3D_R1[101:150], 'b--',
         epoch, PARAFAC4D_R1[101:150], 'm:',
         epoch, PARAFAC4D_R8[101:150], 'm--',
         epoch, Tucker24D_R1[101:150], 'g--',
         epoch, Tucker24D_R8[101:150], 'g:',
         epoch, ATCD4D_R1[101:150], 'o--',
         epoch, ATCD4D_R8[101:150], 'o:')

ax1.legend(['Normal', 'PARAFAC3D R1' ,'ATCD3D R1' ,'PARAFAC4D R1','PARAFAC4D R8','Tucker24D R1','Tucker24D R8', 'ATCD4D R1', 'ATCD4D R8'], loc = 'lower right')
ax2.legend(['Normal', 'PARAFAC3D R1' ,'ATCD3D R1' ,'PARAFAC4D R1','PARAFAC4D R8','Tucker24D R1','Tucker24D R8', 'ATCD4D R1', 'ATCD4D R8'], loc = 'lower right')
ax3.legend(['Normal', 'PARAFAC3D R1' ,'ATCD3D R1' ,'PARAFAC4D R1','PARAFAC4D R8','Tucker24D R1','Tucker24D R8', 'ATCD4D R1', 'ATCD4D R8'], loc = 'lower right')
ax1.set_xlabel('Epochs')
ax1.set_ylim([0.5,1])
ax2.set_xlabel('Epochs')
ax2.set_ylim([0.5,1])
ax3.set_xlabel('Epochs')
#ax2.set_ylim([0.5,1])
ax1.set_ylabel('Training Accuracy')
ax2.set_ylabel('Validation  Accuracy')
ax3.set_ylabel('Loss')
plt.suptitle('Training Results For Methods on conv500 and CIFAR10', fontsize=14)

#plt.savefig('Plot_all_methods_on_mnist_conv4.png')
plt.show()
