import matplotlib.pyplot as plt
import numpy as np

def import_grid(grid_type='esw',res='100'):
    nodes = np.loadtxt('data/'+grid_type+'_nodes_'+res+'k.txt')
    IEN = np.loadtxt('data/'+grid_type+'_IEN_'+res+'k.txt', dtype=np.int64)
    boundary_nodes = np.loadtxt('data/'+grid_type+'_bdry_'+res+'k.txt', 
                                dtype=np.int64)

    plt.triplot(nodes[:,0], nodes[:,1], triangles=IEN)
    plt.plot(nodes[boundary_nodes, 0], nodes[boundary_nodes, 1], 'ro')
    plt.axis('equal')