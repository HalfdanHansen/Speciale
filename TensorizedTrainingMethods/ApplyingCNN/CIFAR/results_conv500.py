import pandas as pd

#Normal
#Rank 1: PARAFAC3D, ATCD3D, PARAFAC4D, Tucker24D, ATCD4D, Tucker2ATCD4D
#Rank 8:  PARAFAC4D, Tucker24D, ATCD4D, Tucker2ATCD4D

#normal = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0905_conv500normalCIFAR10.csv',header=None,index_col=None)
#PARAFAC3D_R1 = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0905_conv500PARAFAC3DCIFAR10.csv',header=None,index_col=None)
#ATCD3D = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0905_conv500ATCD3DCIFAR10.csv',header=None,index_col=None)
#PARAFAC4D_R1 = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0905_conv500PARAFAC4DCIFAR10_rank1.csv',header=None,index_col=None)
#PARAFAC4D_R8 = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0905_conv500PARAFAC4DCIFAR10_rank8.csv',header=None,index_col=None)
#Tucker24D_R1 = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0905_conv500Tucker24DCIFAR10_rank11.csv',header=None,index_col=None)
#Tucker24D_R8 = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0905_conv500Tucker24DCIFAR10_rank88.csv',header=None,index_col=None)
#ATCD4D_R1 = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0905_conv500ATCD4DCIFAR10_rank1.csv',header=None,index_col=None)
#ATCD4D_R8 = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0905_conv500ATCD4DCIFAR10_rank8.csv',header=None,index_col=None)
#Tucker2ATCD4D_R1 = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0905_conv500Tucker2ATCD4DCIFAR10_rank11.csv',header=None,index_col=None)
#Tucker2ATCD4D_R8 = pd.read_csv (r'/zhome/d8/b/107547/Documents/Git/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR/0905_conv500Tucker2ATCD4DCIFAR10_rank88.csv',header=None,index_col=None)

normal = pd.read_csv (r'0605_conv500normalCIFAR10.csv',header=None,index_col=None)
PARAFAC3D_R1 = pd.read_csv (r'0605_conv500PARAFAC3DCIFAR10.csv',header=None,index_col=None)
ATCD3D_R1 = pd.read_csv (r'0505_conv500ATCD3DCIFAR10_rank1.csv',header=None,index_col=None)
PARAFAC4D_R1 = pd.read_csv (r'0605_conv500PARAFAC4DCIFAR10_rank1.csv',header=None,index_col=None)
PARAFAC4D_R8 = pd.read_csv (r'0605_conv500PARAFAC4DCIFAR10_rank8.csv',header=None,index_col=None)
Tucker24D_R1 = pd.read_csv (r'0605_conv500Tucker24DCIFAR10_rank11.csv',header=None,index_col=None)
Tucker24D_R8 = pd.read_csv (r'0605_conv500Tucker24DCIFAR10_rank88.csv',header=None,index_col=None)
ATCD4D_R1 = pd.read_csv (r'0505_conv500ATCD4DCIFAR10_rank1.csv',header=None,index_col=None)
ATCD4D_R8 = pd.read_csv (r'0605_conv500ATCD4DCIFAR10_rank8.csv',header=None,index_col=None)
Tucker2ATCD4D_R1 = pd.read_csv (r'0605_conv500Tucker2ATCD4DCIFAR10_rank11.csv',header=None,index_col=None)
Tucker2ATCD4D_R8 = pd.read_csv (r'0605_conv500Tucker2ATCD4DCIFAR10_rank88.csv',header=None,index_col=None)


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
epoch = np.linspace(0,49,50, dtype = int)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15,6), constrained_layout = True)

l11 = ax1.plot(epoch, normal[0:50], 'r-')
l12 = ax1.plot(epoch, PARAFAC3D_R1[0:50], 'b--')
l13 = ax1.plot(epoch, ATCD3D[0:50], 'g--')
l14 = ax1.plot(epoch, PARAFAC4D_R1[0:50], 'b:')
l15 = ax1.plot(epoch, PARAFAC4D_R8[0:50], 'b-.')
l16 = ax1.plot(epoch, ATCD4D_R1[0:50], 'g:')
l17 = ax1.plot(epoch, ATCD4D_R8[0:50], 'g-.')
l18 = ax1.plot(epoch, Tucker24D_R1[0:50], '#653700', linestyle = ':')
l19 = ax1.plot(epoch, Tucker24D_R8[0:50], '#653700', linestyle = '-.')
l110 = ax1.plot(epoch, Tucker2ATCD4D_R1[50:100], '#808080', linestyle = ':')
l111 = ax1.plot(epoch, Tucker2ATCD4D_R8[50:100], '#808080', linestyle = '-.')

l21 = ax2.plot(epoch, normal[50:100], 'r-')
l22 = ax2.plot(epoch, PARAFAC3D_R1[50:100], 'b--')
l23 = ax2.plot(epoch, ATCD3D[50:100], 'g--')
l24 = ax2.plot(epoch, PARAFAC4D_R1[50:100], 'b:')
l25 = ax2.plot(epoch, PARAFAC4D_R8[50:100], 'b-.')
l26 = ax2.plot(epoch, ATCD4D_R1[50:100], 'g:')
l27 = ax2.plot(epoch, ATCD4D_R8[50:100], 'g-.')
l28 = ax2.plot(epoch, Tucker24D_R1[50:100], '#653700', linestyle = ':')
l29 = ax2.plot(epoch, Tucker24D_R8[50:100], '#653700', linestyle = '-.')
l210 = ax2.plot(epoch, Tucker2ATCD4D_R1[50:100], '#808080', linestyle = ':')
l211 = ax2.plot(epoch, Tucker2ATCD4D_R8[50:100], '#808080', linestyle = '-.')

l31 = ax3.plot(epoch, normal[100:150], 'r-')
l32 = ax3.plot(epoch, PARAFAC3D_R1[100:150], 'b--')
l33 = ax3.plot(epoch, ATCD3D[100:150], 'g--')
l34 = ax3.plot(epoch, PARAFAC4D_R1[100:150], 'b:')
l35 = ax3.plot(epoch, PARAFAC4D_R8[100:150], 'b-.')
l36 = ax3.plot(epoch, ATCD4D_R1[100:150], 'g:')
l37 = ax3.plot(epoch, ATCD4D_R8[100:150], 'g-.')
l38 = ax3.plot(epoch, Tucker24D_R1[100:150], '#653700', linestyle = ':')
l39 = ax3.plot(epoch, Tucker24D_R8[100:150], '#653700', linestyle = '-.')
l310 = ax3.plot(epoch, Tucker2ATCD4D_R1[100:150], '#808080', linestyle = ':')
l311 = ax3.plot(epoch, Tucker2ATCD4D_R8[100:150], '#808080', linestyle = '-.')


ax1.set_xlabel('Epochs')
ax1.set_ylim([0,1])
ax2.set_xlabel('Epochs')
ax2.set_ylim([0,1])
ax3.set_xlabel('Epochs')
ax3.set_ylim([700,2000])
ax1.set_ylabel('Training Accuracy')
ax2.set_ylabel('Validation  Accuracy')
ax3.set_ylabel('Loss')

line_labels = ['Normal', 'CD 3D', 'ATCD 3D', 'CD 4D R1','CD 4D R8','ATCD 4D R1','ATCD 4D R8', 'Tucker2 R1', 'Tucker2 R8', 'Tucker2 ATCD R1', 'Tucker2 ATCD R8']
plt.suptitle('Training Results For Methods on conv500 and CIFAR10', fontsize=20)
fig.tight_layout(pad=2.0)
fig.legend([l11, l12, l13, l14, l15, l16, l17, l18, l19, l110, l111], 
           labels = line_labels,
           loc = 'center right',
           fontsize = 16,
           borderaxespad = 0.1)
plt.subplots_adjust(right=0.82)

plt.savefig('1705_Plot_all_methods_on_cifar10_conv500.pgf')
plt.show()

