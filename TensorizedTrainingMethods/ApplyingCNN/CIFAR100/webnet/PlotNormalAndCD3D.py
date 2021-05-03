#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 09:25:53 2021

@author: s154179
"""

import pandas as pd

datanormal = pd.read_csv('/zhome/ab/6/109248/Desktop/Speciale/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR100/webnet/3004_webNet_deeper.csv', index_col = False, header = None)
data3d = pd.read_csv('/zhome/ab/6/109248/Desktop/Speciale/Speciale/TensorizedTrainingMethods/ApplyingCNN/CIFAR100/webnet/3004_webNet_deeper3d.csv', index_col = False, header = None)

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(0, 50, 1)
Y = [11, 20, 29, 38, 47, 56, 65, 74, 83, 92]
X, Y = np.meshgrid(X, Y)
Z = datanormal.iloc[0:10,:]

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap='Blues',
                       linewidth=1, antialiased=True)

Z = data3d.iloc[0:10,:]

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap='Reds',
                       linewidth=1, antialiased=True)

ax.set_zlim(0, 1)
ax.zaxis.set_major_locator(LinearLocator(10))

ax.zaxis.set_major_formatter('{x:.1f}')


plt.show()