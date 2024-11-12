import matplotlib.pyplot as plt
import numpy as np

nodes = np.loadtxt('data/esw_nodes_100k.txt')
IEN = np.loadtxt('data/esw_IEN_100k.txt', dtype=np.int64)
boundary_nodes = np.loadtxt('data/esw_bdry_100k.txt', 
                            dtype=np.int64)

plt.triplot(nodes[:,0], nodes[:,1], triangles=IEN)
plt.plot(nodes[boundary_nodes, 0], nodes[boundary_nodes, 1], 'ro')
plt.axis('equal')