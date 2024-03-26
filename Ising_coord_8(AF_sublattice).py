import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
from scipy.spatial import cKDTree
import time
import sys
import pandas as pd
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
def sublattice(lattice):
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

    return np.array(a), np.array(b)

@njit
def sublattice_mag(grid):
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
frac = 0.75
a = 1
lattice = ising_lattice(N, a)
neighbor_list = neighbors(lattice, a)

J = -1
T=np.linspace(0.1,3.5,50)
steps=10000
eqsteps=2000
Magnetization_A = []
Magnetization_B = []
Magnetization_diff = []

for t in T:
    M_A = 0
    M_B = 0
    grid = initial(N, frac)

    for k in range(eqsteps):
        flip(grid, neighbor_list, t, J, N)
    for l in range(steps):
        flip(grid, neighbor_list, t, J, N)

        grid_a, grid_b = sublattice_mag(grid)

        m_a = sum(grid_a)/((N**2)/2)
        #m_a = abs(sum(grid_a)/((N**2)/2))
        m_b = sum(grid_b)/((N**2)/2)
        #m_b = abs(sum(grid_b)/((N**2)/2))
        M_A += m_a
        M_B += m_b

    Mag_A = M_A/steps
    Mag_B = M_B/steps

    Magnetization_A.append(Mag_A)
    Magnetization_B.append(Mag_B)
    Magnetization_diff.append(Mag_A + Mag_B)
    #Magnetization_diff.append(Mag_A - Mag_B)

plt.figure(figsize=(8,10),dpi=200)
#plt.figure()
plt.subplot(311)
plt.plot(T,Magnetization_A,color='red', marker='.')
plt.xlabel('Temperature ($J/k_b$)')
plt.ylabel('Average Magnetization per spin')
plt.title(f'$M_A$ vs Temperature (N={N})')
#plt.title(f'$|M_A|$ vs Temperature (N={N})')
plt.grid()
plt.tight_layout()
#plt.figure()
plt.subplot(312)
plt.plot(T,Magnetization_B,color='teal', marker='.')
plt.xlabel('Temperature ($J/k_b$)')
plt.ylabel('Average Magnetization per spin')
plt.title(f'$M_B$ vs Temperature (N={N})')
#plt.title(f'$|M_B|$ vs Temperature (N={N})')
plt.grid()
plt.tight_layout()
plt.subplot(313)
plt.plot(T,Magnetization_diff,color='black', marker='.')
plt.xlabel('Temperature ($J/k_b$)')
plt.ylabel('Average Magnetization per spin')
plt.title(f'$M_A$ + $M_B$ vs Temperature (N={N})')
#plt.title(f'$|M_A|$ - $|M_B|$ vs Temperature (N={N})')
plt.grid()
plt.tight_layout()
f_time=time.time()
e_time=f_time-s_time
print(f'Execution time: {e_time:.2f} seconds')

#plt.savefig(f'D:\Desktop\Ising Model\Ising_code\Ising-coord-AF_N{N}_f{frac}_10e4+2e3_sublattice.png')

plt.show()