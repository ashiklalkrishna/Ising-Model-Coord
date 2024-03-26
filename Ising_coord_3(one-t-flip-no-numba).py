import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
from scipy.spatial import cKDTree
import time
import sys

s_time=time.time()

def ising_lattice (N, a):
    pos = np.zeros((N**2, 2), dtype=float)
    y=0
    i=0
    nrows = N
    ncols = N

    for row in range(0, nrows):
        x=0
        for col in range (0, ncols):
            pos[i] = [x, y]
            x += a

            i += 1
        y += a
    
    return pos

def initial(N, frac):
    rand_grid=np.random.random(N**2)
    ini_grid=np.zeros(N**2)
    ini_grid[rand_grid>=frac] = 1
    ini_grid[rand_grid<frac] = -1
    return ini_grid

def neighbors(lattice, j, a):
    tree = cKDTree(lattice)

    query_point = lattice[j]
    distance_threshold = a + 0.1
    nearest_neighbors_indices = tree.query_ball_point(query_point, distance_threshold)
    nearest_neighbors_indices.remove(j)

    return nearest_neighbors_indices

#@njit("f8[:](f8[:,:],f8[:], f8, f8, i8, f8)")
def flip(lattice, grid, t, J, N, a):   
    for k in range(0,N):
        for l in range(0,N):
            i = np.random.randint(0,N**2)  
            p = lattice[i]    
            s = grid[i]  

            nearest_neighbors_indices = neighbors(lattice, i, a)
            neighbor_spin = []
            for j in nearest_neighbors_indices:
                neighbor_spin.append(grid[j])
            nn = sum(neighbor_spin)

            de = 2*J*s*nn
            s_flip=-1*s                
      
            if de<=0:
                s = s_flip
            elif np.random.random() < np.exp(-de/t):
                s = s_flip
        
            grid[i] = s
    return grid

#@njit("f8(f8[:,:], f8[:], f8, i8, f8)") 
def energy_func(lattice, grid, J, N, a):
    e=0
    for k in range(0,N):
        for l in range(0,N):
            i = np.random.randint(0,N)  
            p = lattice[i]  
            s = grid[i]  

            nearest_neighbors_indices = neighbors(lattice, a, i)
            neighbor_spin = []
            for i in nearest_neighbors_indices:
                neighbor_spin.append(grid[i])
            nn = sum(neighbor_spin)

            e += -J*s*nn
    return e/2

N = 8
frac = 0.5
a = 1
lattice = ising_lattice(N, a)
grid = initial(N, frac)
ini_grid = grid.copy()

'''fig = plt.figure()
scatter = plt.scatter(lattice[:, 0], lattice[:, 1], c=ini_grid, cmap='viridis', marker='s', s=50)
cbar = plt.colorbar(scatter)
cbar.set_label('State')
#for i, txt in enumerate(range(len(lattice))):
    #plt.text(lattice[i, 0], lattice[i, 1], str(i), fontsize=8, ha='center', va='center')'''

J=1
t=3.222
steps=1000
eqsteps=100
M=0

for k in range(eqsteps):
    flip(lattice, grid, t, J, N, a)
for l in range(steps):
    flip(lattice, grid, t, J, N, a)

plt.figure(figsize=(10,5))
plt.subplot(121)
scatter = plt.scatter(lattice[:, 0], lattice[:, 1], c=ini_grid, cmap='viridis', marker='s', s=50)
cbar = plt.colorbar(scatter)
cbar.set_label('State')
plt.tight_layout()
plt.subplot(122)
scatter = plt.scatter(lattice[:, 0], lattice[:, 1], c=grid, cmap='viridis', marker='s', s=50)
cbar = plt.colorbar(scatter)
cbar.set_label('State')
plt.tight_layout()

plt.show()