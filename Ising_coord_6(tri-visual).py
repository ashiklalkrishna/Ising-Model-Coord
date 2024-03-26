import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
from scipy.spatial import cKDTree
import time
import sys

import pandas as pd
s_time=time.time()

def tri_lattice (N, a):
    pos = np.zeros((N**2, 2), dtype=float)

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
            pos[i] = [x, y]
            x += a
            i += 1
        y += a * np.sqrt(3) / 2
    
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
    neigh_edge_l = []
    neigh_edge_corn_r = []
    neigh_edge_tb = []
    neigh_corn_l = []
    sorted_indices = []

    tree = cKDTree(lattice)
    distance_threshold = a + 0.2
    
    for i in range(N**2):
        query_point = lattice[i]
        nearest_neighbors_indices = tree.query_ball_point(query_point, distance_threshold)
        nearest_neighbors_indices.remove(i)

        nearest_neighbors_indices = np.insert (nearest_neighbors_indices, 0, i)

        if len(nearest_neighbors_indices) == 7:
            neigh_bulk.append(nearest_neighbors_indices)
        elif len(nearest_neighbors_indices) == 6:
            nearest_neighbors_indices = np.append(nearest_neighbors_indices, -1)
            neigh_edge_l.append(nearest_neighbors_indices)
        elif len(nearest_neighbors_indices) == 4:
            nearest_neighbors_indices = np.append(nearest_neighbors_indices, -1)
            nearest_neighbors_indices = np.append(nearest_neighbors_indices, -1)
            nearest_neighbors_indices = np.append(nearest_neighbors_indices, -1)
            neigh_edge_corn_r.append(nearest_neighbors_indices)  
        elif len(nearest_neighbors_indices) == 5:
            nearest_neighbors_indices = np.append(nearest_neighbors_indices, -1)
            nearest_neighbors_indices = np.append(nearest_neighbors_indices, -1)
            neigh_edge_tb.append(nearest_neighbors_indices)       
        elif len(nearest_neighbors_indices) == 3:
            nearest_neighbors_indices = np.append(nearest_neighbors_indices, -1)
            nearest_neighbors_indices = np.append(nearest_neighbors_indices, -1)
            nearest_neighbors_indices = np.append(nearest_neighbors_indices, -1)
            nearest_neighbors_indices = np.append(nearest_neighbors_indices, -1)
            neigh_corn_l.append(nearest_neighbors_indices)
            
    neighbor_list = neigh_bulk + neigh_edge_l + neigh_edge_tb + neigh_edge_corn_r + neigh_corn_l

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
            for i in range(1,7):
                neigh = neighbor_list[indx][i]
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
            for i in range(1,7):
                neigh = neighbor_list[indx][i]
                if neigh != None:
                    nn += grid[neigh]

            e += -J*s*nn
    return e/3

N = 16
J = -1
frac = 0.75
a = 1
lattice = tri_lattice(N, a)
neighbor_list = neighbors(lattice, a)

T=np.linspace(0.1,2,10)
steps=1000
eqsteps=200


'''#single t flip
t = 0.01
grid = initial(N, frac)
for k in range(eqsteps):
    flip(grid, neighbor_list, t, J, N)
for l in range(steps):
    flip(grid, neighbor_list, t, J, N)
fig = plt.figure()
scatter = plt.scatter(lattice[:, 0], lattice[:, 1], c=grid, cmap='viridis', marker='s', s=100)
plt.title(f'T = {t}')
cbar = plt.colorbar(scatter)
cbar.set_label('State')
#plt.savefig(r'D:\Desktop\Ising Model\Ising_code\Ising-coord_N256_f0.75_10e3-visual.png')
plt.show()'''

'''
#multi t flip
for t in T:
    grid = initial(N, frac)
    for k in range(eqsteps):
        flip(grid, neighbor_list, t, J, N)
    for l in range(steps):
        flip(grid, neighbor_list, t, J, N)
    fig = plt.figure()
    scatter = plt.scatter(lattice[:, 0], lattice[:, 1], c=grid, cmap='viridis', marker='s', s=100)
    plt.title(f'T = {t}')
    cbar = plt.colorbar(scatter)
    cbar.set_label('State')
    #plt.savefig(f'D:\Desktop\Ising Model\Ising_code\Ising-coord_N{N}_f0.75_10e3{t}.png')
'''

#single t multi iteration
t=0.01
for i in range(25):
    grid = initial(N, frac)
    for k in range(eqsteps):
        flip(grid, neighbor_list, t, J, N)
    for l in range(steps):
        flip(grid, neighbor_list, t, J, N)
    fig = plt.figure(dpi=200)
    scatter = plt.scatter(lattice[:, 0], lattice[:, 1], c=grid, cmap='viridis', marker='s', s=100)
    plt.title(f'T = {t}, Frame {i}')
    cbar = plt.colorbar(scatter)
    cbar.set_label('State')
    plt.savefig(f'D:\Desktop\Ising Model\Ising_code\Animation\Ising-coord_N{N}_f0.75_10e3_{i}.png')
    plt.close()

f_time=time.time()
e_time=f_time-s_time
print(f'Execution time: {e_time:.2f} seconds')