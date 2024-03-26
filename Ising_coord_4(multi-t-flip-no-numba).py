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

def neighbors(lattice, tree, j, a):

    query_point = lattice[j]
    distance_threshold = a + 0.1
    nearest_neighbors_indices = tree.query_ball_point(query_point, distance_threshold)
    nearest_neighbors_indices.remove(j)

    return nearest_neighbors_indices

#@njit("f8[:](f8[:,:],f8[:], f8, f8, i8, f8)")
def flip(lattice, grid, tree, t, J, N, a):   
    for k in range(0,N):
        for l in range(0,N):
            i = np.random.randint(0,N**2)  
            p = lattice[i]    
            s = grid[i]  

            nearest_neighbors_indices = neighbors(lattice, tree, i, a)
            neighbor_spin = []
            for j in nearest_neighbors_indices:
                neighbor_spin.append(grid[j])
            nn = sum(neighbor_spin)

            de = 2*J*s*nn
            s_flip = -1*s                
      
            if de<=0:
                s = s_flip
            elif np.random.random() < np.exp(-de/t):
                s = s_flip
        
            grid[i] = s
    return grid

#@njit("f8(f8[:,:], f8[:], f8, i8, f8)") 
def energy_func(lattice, grid, tree, J, N, a):
    e=0
    for k in range(0,N):
        for l in range(0,N):
            i = np.random.randint(0,N)  
            p = lattice[i]  
            s = grid[i]  

            nearest_neighbors_indices = neighbors(lattice, tree, a, i)
            neighbor_spin = []
            for i in nearest_neighbors_indices:
                neighbor_spin.append(grid[i])
            nn = sum(neighbor_spin)
            e += -J*s*nn
    return e/2

N = 8
frac = 0.75
a = 1
lattice = ising_lattice(N, a)
tree = cKDTree(lattice)
J=1
T=np.linspace(0.5,4,50)
steps=1000
eqsteps=100
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
        flip(lattice, grid, tree, t, J, N, a)
    for l in range(steps):
        flip(lattice, grid, tree, t, J, N, a)
        e = energy_func(lattice, grid, tree, J, N, a)/(N**2)
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

#plt.savefig(r'D:\Desktop\Ising Model\Ising_code\isng_test.png')
plt.show()