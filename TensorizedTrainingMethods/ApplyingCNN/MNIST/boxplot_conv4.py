import os 
import sys 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


N = pd.read_csv(r'MNIST_conv4_normal.csv',header=None,index_col=None)
CD3D = pd.read_csv(r'MNIST_conv4_CD3D.csv',header=None,index_col=None)
BAF3D = pd.read_csv(r'MNIST_conv4_BAF3D.csv',header=None,index_col=None)
ATCD3D = pd.read_csv(r'MNIST_conv4_ATCD3D.csv',header=None,index_col=None)
CD4D = pd.read_csv(r'MNIST_conv4_CD4D.csv',header=None,index_col=None)
BAF4D = pd.read_csv(r'MNIST_conv4_BAF4D.csv',header=None,index_col=None)
ATCD4D = pd.read_csv(r'MNIST_conv4_ATCD4D.csv',header=None,index_col=None)

#Bdf = pd.DataFrame(columns = ["BL", "3DCD", "3DBAF", "3DATCD", "4DCD", "4DBAF", "4DATCD"])

#cols = ["BL", "3DCD", "3DBAF", "3DATCD", "4DCD", "4DBAF", "4DATCD"]

Nt = N[49][0:10]
C3t = CD3D[49][0:10]
B3t = BAF3D[49][0:10]
A3t = ATCD3D[49][0:10]
C4t = CD4D[49][0:10]
B4t = BAF4D[49][0:10]
A4t = ATCD4D[49][0:10]
Nv = N[49][10:20]
C3v = CD3D[49][10:20]
B3v = BAF3D[49][10:20]
A3v = ATCD3D[49][10:20]
C4v = CD4D[49][10:20]
B4v = BAF4D[49][10:20]
A4v = ATCD4D[49][10:20]

B = pd.DataFrame(columns = ["Train", "Val", "Method"])

m = [["BL"]*10, ["3DCD"]*10, ["3DBAF"]*10, ["3DATCD"]*10, ["4DCD"]*10, ["4DBAF"]*10, ["4DATCD"]*10]

flatm = [item for sublist in m for item in sublist]

B.Train = pd.concat([Nt,C3t,B3t,A3t,C4t,B4t,A4t], ignore_index=True)
B.Val = pd.concat([Nv,C3v,B3v,A3v,C4v,B4v,A4v], ignore_index=True)
B.Method = flatm


clist = ["r", "b", "m", "g", "b", "m", "g"]
llist = ["-", "--", "--", "--", ":", ":", ":"]

#fig, axes = plt.subplots(1,2, figsize=(30,10))

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(11,6))

ax1.set_ylabel('Training Accuracy')
ax2.set_ylabel('Validation  Accuracy')
ax2.set_xlabel('Epochs')
plt.suptitle('Boxplots for conv4 with 50 epochs, MNIST', fontsize=20)

sns.boxplot(y='Train', x='Method', 
                 data=B, 
                 width=0.5,
                 palette=clist,
                 ax = ax1
                 ).set( 
    ylabel='Training Accuracy'
)
sns.boxplot(y='Val', x='Method', 
                 data=B, 
                 width=0.5,
                 palette=clist,
                 ax = ax2
                 ).set( 
    ylabel='Validation Accuracy'
)

plt.savefig('boxplot.pgf')
plt.show()

