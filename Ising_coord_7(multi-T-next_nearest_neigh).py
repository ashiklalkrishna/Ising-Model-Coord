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

def near_neighbors(lattice):
    neighbor_list = []
    sorted_indices = []

    tree = cKDTree(lattice)
    distance_threshold = np.inf + 0.1
    
    for i in range(N**2):
        query_point = lattice[i]
        neighbors_indices = tree.query_ball_point(query_point, distance_threshold)
        neighbors_indices.remove(i)

        neighbors_indices = np.insert (neighbors_indices, 0, i)
        neighbor_list.append(neighbors_indices)
    neighbor_list = np.array(neighbor_list)

    #sorted_indices = np.argsort(neighbors_indices[:, 0])
    #neighbor_list = neighbor_list[sorted_indices]

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

N = 64
frac = 0.9
a = 1
lattice = ising_lattice(N, a)
neighbor_list = near_neighbors(lattice)

J = 1
T=np.linspace(0.5,3.5,50)
steps=1000
eqsteps=200
energy=[]
magnetization=[]
specific_heat=[]
susceptibility=[]

for t in T:
    E=0
    M=0
    E_sqr=0
    M_sqr=0
    grid = initial(N, frac)
    for k in range(eqsteps):
        flip(grid, neighbor_list, t, J, N)
    for l in range(steps):
        flip(grid, neighbor_list, t, J, N)
        e = energy_func(grid, neighbor_list, J, N)/(N**2)
        m = sum(grid)/(N**2)
        E += e
        M += m
        E_sqr += e**2
        M_sqr += m**2
    
    E_mean = E/steps
    M_mean = M/steps
    E_sqr_mean = E_sqr/steps
    M_sqr_mean = M_sqr/steps
    
    Cv = (E_sqr_mean -  E_mean**2)/((t**2))
    X = (M_sqr_mean - M_mean**2)/(t)
     
    energy.append(E_mean)
    magnetization.append(M_mean)
    specific_heat.append(Cv)
    susceptibility.append(X)

plt.figure(figsize=(8,15))
plt.subplot(411)
#plt.figure()
plt.plot(T,energy,color='orangered', marker='.')
plt.xlabel('Temperature ($J/k_b$)')
plt.ylabel('Average Energy per Spin')
plt.title(f'Average Energy vs Temperature (N={N})')
plt.grid()

#plt.figure()
plt.subplot(412)
plt.plot(T,magnetization,color='teal', marker='.')
plt.xlabel('Temperature ($J/k_b$)')
plt.ylabel('Average Magnetization per spin')
plt.title(f'Average Magnetization vs Temperature (N={N})')
plt.grid()

#plt.figure()
plt.subplot(413)
plt.plot(T,specific_heat,color='maroon', marker='.')
plt.xlabel('Temperature ($J/k_b$)')
plt.ylabel('Specific heat')
plt.title(f'Specific heat vs Temperature (N={N})')
plt.grid()

#plt.figure()
plt.subplot(414)
plt.plot(T,susceptibility,color='darkblue', marker='.')
plt.xlabel('Temperature ($J/k_b$)')
plt.ylabel('Susceptibility')
plt.title(f'Susceptibility vs Temparature (N={N})')
plt.grid()
plt.tight_layout()

f_time=time.time()
e_time=f_time-s_time
print(f'Execution time: {e_time:.2f} seconds')

plt.savefig(f'D:\Desktop\Ising Model\Ising_code\Ising-coord_N{N}_f0.75_10e3_inf_absM.png')
plt.show()