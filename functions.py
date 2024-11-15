"""Choose a way of modelling the pollutant spread.
    Choose a method to numerically solve your model on an unstructured grid.
    Write a code to solve the model using the chosen method.
    Find how much of the pollutant passed over the University of Reading, using some of the provided grids.
    Write a report of no more than ten pages (not including references and appendices) explaining your results and their accuracy, both in terms of modelling and numerics."""

import matplotlib.pyplot as plt
import numpy as np

#coordinates
UoS = [442365, 115483]
UoR = [473993, 171625]

def import_grid(grid_type='esw',res='100'):
    nodes = np.loadtxt('data/'+grid_type+'_nodes_'+res+'k.txt')
    IEN = np.loadtxt('data/'+grid_type+'_IEN_'+res+'k.txt', dtype=np.int64)
    boundary_nodes = np.loadtxt('data/'+grid_type+'_bdry_'+res+'k.txt', 
                                dtype=np.int64)

    plt.triplot(nodes[:,0], nodes[:,1], triangles=IEN)
    plt.plot(nodes[boundary_nodes, 0], nodes[boundary_nodes, 1], 'ro')
    plt.axis('equal')

def model():
    """Equations
        diffusion equation eg
        The minimal model would solve for the pollutant as a scalar field indicating the pollutant concentration normalized to one over Southampton. 
        This could be done using a time independent or time dependent model. The velocity field could be imposed (in the simplest model) or solved for (much more complex). 
        A reasonable approximation to the wind conditions that day would be a constant 10 metre per second wind to the north.

        Source or boundary conditions
        The pollutant was sourced by the fire at a very specific and narrow location. 
        However, on all practical grids this source will be within a single element, and that element will be very close to the boundary. 
        The pollutant can therefore be injected into the domain either using a localised source function (local in space for stationary models; 
        for time dependent models, the fire lasted about 8 hours) or by setting Dirichlet boundary conditions over some relevant boundary nodes."""
    
    return None

def method():
    """In this section we have seen essentially three methods that would work on unstructured meshes;
        1. time independent Galerkin finite element solutions of diffusion equations (advection terms can be added, with care);
        2. time dependent Galerkin finite element solutions of advection equations (diffusion terms can be added, with care);
        3. time dependent Discontinuous Galerkin finite element solutions of advection equations (adding diffusive terms requires careful work).
        The choice of method should be linked to the model chosen.
        Code from standard modules (eg numpy, scipy, quadpy) can be used, but full PDE solving packages (eg FEniCS, dedalus) cannot."""
    
    return None