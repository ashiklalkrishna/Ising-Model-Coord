import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
import time
from scipy.spatial import cKDTree
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
    
N = 5
frac = 0.01
a = 1
d = 5
lattice = ising_lattice(N, a)
grid = initial (N, frac)

def neighbors(lattice, i, d):
    tree = cKDTree(lattice)

    query_point = lattice[i]
    distance_threshold = d + 0.1
    nearest_neighbors_indices = tree.query_ball_point(query_point, distance_threshold)
    nearest_neighbors_indices.remove(i)

    return nearest_neighbors_indices

neighbour_list = neighbors(lattice, 5, d)
print("Nearest neighbors indices:", neighbour_list)
neighbor_spin = []
for i in neighbour_list:
    neighbor_spin.append(grid[i])
#print("Nearest neighbors spin states:",neighbor_spin)

fig = plt.figure()
scatter = plt.scatter(lattice[:, 0], lattice[:, 1], c=grid, cmap='viridis', marker='s', s=100)
cbar = plt.colorbar(scatter)
cbar.set_label('State')
for i, txt in enumerate(range(len(lattice))):
    plt.text(lattice[i, 0], lattice[i, 1], str(i), fontsize=8, ha='center', va='center')


plt.show()