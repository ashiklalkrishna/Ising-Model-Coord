import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.spatial import cKDTree
import time
import sys
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

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

def neighbors(lattice, a):
    neighbor_list = []
    neigh_bulk = []
    neigh_edge = []
    neigh_corn = []
    sorted_indices = []

    tree = cKDTree(lattice)
    distance_threshold = a + 0.1
    
    for i in range(N**2):
        query_point = lattice[i]
        nearest_neighbors_indices = tree.query_ball_point(query_point, distance_threshold)
        nearest_neighbors_indices.remove(i)

        nearest_neighbors_indices = np.insert (nearest_neighbors_indices, 0, i)

        if len(nearest_neighbors_indices) == 5:
            neigh_bulk.append(nearest_neighbors_indices)
        elif len(nearest_neighbors_indices) == 4:
            nearest_neighbors_indices = np.append(nearest_neighbors_indices, -1)
            neigh_edge.append(nearest_neighbors_indices)        
        elif len(nearest_neighbors_indices) == 3:
            nearest_neighbors_indices = np.append(nearest_neighbors_indices, -1)
            nearest_neighbors_indices = np.append(nearest_neighbors_indices, -1)
            neigh_corn.append(nearest_neighbors_indices)
            
    neighbor_list = neigh_bulk + neigh_edge + neigh_corn
    neighbor_list = np.array(neighbor_list)
    sorted_indices = np.argsort(neighbor_list[:, 0])
    neighbor_list = neighbor_list[sorted_indices]

    return neighbor_list

@njit
def flip(grid, neighbor_list, t, J, N):   
    for k in range(0,N):
        for l in range(0,N):
            indx = np.random.randint(0,N**2) 
            s = grid[indx]  

            nn = 0
            for i in range(1,5):
                neigh = neighbor_list[indx, i]
                if neigh != -1:
                    nn += grid[neigh]

            de = 2*J*s*nn
            s_flip = -1*s                
      
            if de<=0:
                s = s_flip
            elif np.random.random() < np.exp(-de/t):
                s = s_flip
        
            grid[indx] = s
    return grid

@njit
def energy_func(grid, neighbor_list, J, N):
    e=0
    for k in range(0,N):
        for l in range(0,N):
            indx = np.random.randint(0,N**2) 
            s = grid[indx]  

            nn = 0
            for i in range(1,5):
                neigh = neighbor_list[indx][i]
                if neigh != None:
                    nn += grid[neigh]

            e += -J*s*nn
    return e/2


def sublattice(lattice, N):
    a = []
    b = []
    for row in range(N):
        if row % 2 == 0:
            for col in range(N):
                if col % 2 == 0:
                    a.append(lattice[row*N+col])
                else:
                    b.append(lattice[row*N+col])
        else:
            for col in range(N):
                if col % 2 != 0:
                    a.append(lattice[row*N+col])
                else:
                    b.append(lattice[row*N+col])
    a = np.array(a)
    b = np.array(b)
    return a, b


def sublattice_mag(grid, N):
    grid_a = []
    grid_b = []
    for row in range(N):
        if row % 2 == 0:
            for col in range(N):
                if col % 2 == 0:
                    grid_a.append(grid[row*N+col])
                else:
                    grid_b.append(grid[row*N+col])
        else:
            for col in range(N):
                if col % 2 != 0:
                    grid_a.append(grid[row*N+col])
                else:
                    grid_b.append(grid[row*N+col])
    return np.array(grid_a), np.array(grid_b)

N = 8
frac = 0.25
a = 1
lattice = ising_lattice(N, a)
neighbor_list = neighbors(lattice, a)

J = -1
steps=1000
eqsteps=200

'''
#single t sublattice
t = 0.1
grid = initial(N, frac)
for k in range(eqsteps):
    flip(grid, neighbor_list, t, J, N)
for l in range(steps):
    flip(grid, neighbor_list, t, J, N)
    
A, B = sublattice(lattice, N)
grid_a, grid_b = sublattice_mag(grid, N)

plt.figure(figsize=(10,5))
plt.subplot(121)
scatter_a = plt.scatter(A[:, 0], A[:, 1], c=grid_a, cmap='bwr', marker='s', s=100)
plt.title(f'T = {t}')
cbar_a = plt.colorbar(scatter_a)
cbar_a.set_label('State')
scatter_a.set_clim(-1, 1)
for i, txt in enumerate(range(len(lattice))):
    plt.text(lattice[i, 0], lattice[i, 1], str(i), fontsize=8, ha='center', va='center')

plt.subplot(122)
scatter_b = plt.scatter(B[:, 0], B[:, 1], c=grid_b, cmap='bwr', marker='s', s=100)
plt.title(f'T = {t}')
cbar_b = plt.colorbar(scatter_b)
cbar_b.set_label('State')
scatter_b.set_clim(-1, 1)
for i, txt in enumerate(range(len(lattice))):
    plt.text(lattice[i, 0], lattice[i, 1], str(i), fontsize=8, ha='center', va='center')
#plt.savefig(r'D:\Desktop\Ising Model\Ising_code\Ising-coord_N256_f0.75_10e3-visual.png')
plt.tight_layout()
plt.show()
'''

'''
#multi t sublattice
T=np.linspace(0.01,3.5,50)
for t in T:
    grid = initial(N, frac)

    for k in range(eqsteps):
        flip(grid, neighbor_list, t, J, N)
    for l in range(steps):
        flip(grid, neighbor_list, t, J, N)

        A, B = sublattice(lattice, N)
        grid_a, grid_b = sublattice_mag(grid, N)

    plt.figure(figsize=(10,5), dpi=200)
    plt.subplot(121)
    scatter_a = plt.scatter(A[:, 0], A[:, 1], c=grid_a, cmap='viridis', marker='s', s=100)
    plt.title(f'T = {t:.2f}')
    cbar_a = plt.colorbar(scatter_a)
    cbar_a.set_label('State')
    scatter_a.set_clim(-1, 1)
    for i, txt in enumerate(range(len(lattice))):
        plt.text(lattice[i, 0], lattice[i, 1], str(i), fontsize=8, ha='center', va='center')

    plt.subplot(122)
    scatter_b = plt.scatter(B[:, 0], B[:, 1], c=grid_b, cmap='viridis', marker='s', s=100)
    plt.title(f'T = {t:.2f}')
    cbar_b = plt.colorbar(scatter_b)
    cbar_b.set_label('State')
    scatter_b.set_clim(-1, 1)
    for i, txt in enumerate(range(len(lattice))):
        plt.text(lattice[i, 0], lattice[i, 1], str(i), fontsize=8, ha='center', va='center')
    plt.tight_layout()
    plt.savefig(f'D:\Desktop\Ising Model\Ising_code\Animation\Ising-coord-AF_N{N}_f{frac}_10e3-visual{t:.2f}.png')
    plt.close()
'''
'''
#single t iteration
t =0.01
for p in range(0, 30):
    grid = initial(N, frac)

    for k in range(eqsteps):
        flip(grid, neighbor_list, t, J, N)
    for l in range(steps):
        flip(grid, neighbor_list, t, J, N)

        A, B = sublattice(lattice, N)
        grid_a, grid_b = sublattice_mag(grid, N)

    plt.figure(figsize=(10,5), dpi=200)
    plt.subplot(121)
    scatter_a = plt.scatter(A[:, 0], A[:, 1], c=grid_a, cmap='viridis', marker='s', s=100)
    plt.title(f'T = {t:.3f}, Frame {p}')
    cbar_a = plt.colorbar(scatter_a)
    cbar_a.set_label('State')
    scatter_a.set_clim(-1, 1)
    for i, txt in enumerate(range(len(lattice))):
        plt.text(lattice[i, 0], lattice[i, 1], str(i), fontsize=8, ha='center', va='center')

    plt.subplot(122)
    scatter_b = plt.scatter(B[:, 0], B[:, 1], c=grid_b, cmap='viridis', marker='s', s=100)
    plt.title(f'T = {t:.3f}, Frame {p}')
    cbar_b = plt.colorbar(scatter_b)
    cbar_b.set_label('State')
    scatter_b.set_clim(-1, 1)
    for i, txt in enumerate(range(len(lattice))):
        plt.text(lattice[i, 0], lattice[i, 1], str(i), fontsize=8, ha='center', va='center')
    plt.tight_layout()
    #plt.savefig(f'D:\Desktop\Ising Model\Ising_code\Animation\Ising-coord-AF_N{N}_f{frac}_10e3-visual_T{t}_({p}).png')
    plt.close()
'''