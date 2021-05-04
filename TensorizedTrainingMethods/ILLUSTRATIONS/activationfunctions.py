import numpy as np
import matplotlib.pyplot as plt

def RELU(x):
    ''' It returns zero if the input is less than zero otherwise it returns the given input. '''
    x1=[]
    for i in x:
        if i<0:
            x1.append(0)
        else:
            x1.append(i)

    return x1

def tanh(x):
    ''' It returns the value (1-exp(-2x))/(1+exp(-2x)) and the value returned will be lies in between -1 to 1.'''
    return np.tanh(x)

def sigmoid(x):
    ''' It returns 1/(1+exp(-x)). where the values lies between zero and one '''
    return 1/(1+np.exp(-x))

def linear(x):
    ''' y = f(x) It returns the input as it is'''
    return x

fig, axs = plt.subplots(1,4,figsize=(12,4))
fig.suptitle('Activation Functions',fontsize=20, y = 0.9999)

plt.setp(axs,yticks=[])

plt.sca(axs[0])
plt.yticks(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))

for ax in axs:
    ax.set_ylim([-1.3,1.3])

x = np.linspace(-1.3, 1.3)
axs[0].plot(x, linear(x))
axs[0].set_title('Linear',fontsize=10)
axs[0].set_xlim([-1.3, 1.3])
plt.sca(axs[0])
plt.xticks([-1, 0, 1], [-1, 0, 1])


x = np.linspace(-3.3, 3.3)
axs[1].plot(x, sigmoid(x))
axs[1].set_title('Sigmoid',fontsize=10)
axs[1].set_xlim([-3.3, 3.3])
plt.sca(axs[1])
plt.xticks([-3,-2,-1,0,1,2,3], [-3,-2,-1,0,1,2,3])

x = np.linspace(-3.3, 3.3)
axs[2].plot(x, tanh(x))
axs[2].set_title('Tanh',fontsize=10)
axs[2].set_xlim([-3.3, 3.3])
plt.sca(axs[2])
plt.xticks([-3,-2,-1,0,1,2,3], [-3,-2,-1,0,1,2,3])

x = np.linspace(-1.3, 1.3)
axs[3].plot(x, RELU(x))
axs[3].set_title('RELU',fontsize=10)
axs[3].set_xlim([-1.3, 1.3])
plt.sca(axs[3])
plt.xticks([-1,0,1], [-1,0,1])

