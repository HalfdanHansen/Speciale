import pandas as pd

datanormal = pd.read_csv('3004_webNet_deeper.csv', index_col = False, header = None)
data3d = pd.read_csv('3004_webNet_deeper3d.csv', index_col = False, header = None)

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

%matplotlib inline
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(0, 50, 1)
Y = [11, 20, 29, 38, 47, 56, 65, 74, 83, 92]
X, Y = np.meshgrid(X, Y)
Z = datanormal.iloc[0:10,:]

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap='Reds',
                       linewidth=1, antialiased=True)

Z = data3d.iloc[0:10,:]

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap='Blues',
                       linewidth=1, antialiased=False)

ax.set_zlim(0, .7)
ax.zaxis.set_major_locator(LinearLocator(10))

ax.zaxis.set_major_formatter('{x:.1f}')

plt.show()