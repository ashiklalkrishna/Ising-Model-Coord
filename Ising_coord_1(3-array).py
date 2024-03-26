import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
from scipy.spatial import cKDTree
import time

s_time=time.time()

def ising_lattice (N, frac, a):
    pos = np.zeros((N**2, 3), dtype=float)

    y=0
    i=0
    nrows = N
    ncols = N

    for row in range(0, nrows):
        x=0
        for col in range (0, ncols):

            random = np.random.random()
            if random >= frac:
                spin = +1
            else: 
                spin = -1

            pos[i] = [x, y, spin]
            x += a
            i += 1
        y += a
    
    return pos

#triangular lattice
def tri_lattice (N, frac):
    pos = np.zeros((N**2, 3), dtype=float)

    y=0
    i=0
    nrows = N
    ncols = N

    for row in range(0, nrows):
        if row % 2 == 0:
            x = 0
        else:
            x = 0.5
        for col in range (0, ncols):

            random = np.random.random()
            if random >= frac:
                spin = +1
            else: 
                spin = -1

            pos[i] = [x, y, spin]
            x += a
            i += 1
        y += a * np.sqrt(3) / 2
    
    return pos

N = 10
frac = 0.25
a = 1
grid = ising_lattice(N, frac, a)

fig = plt.figure()
scatter = plt.scatter(grid[:, 0], grid[:, 1], c=grid[:, 2], cmap='viridis', marker='s', s=50)
cbar = plt.colorbar(scatter)
cbar.set_label('State')

for i, txt in enumerate(range(len(grid))):
    plt.text(grid[i, 0], grid[i, 1], str(i), fontsize=8, ha='center', va='center')

plt.show()