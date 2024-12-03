import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse as sp

def shape_functions(xi):
    return np.array([1-xi[0]-xi[1], xi[0], xi[1]])

def div_xi_shape_functions():
    return np.array([[-1,1,0],[-1,0,1]])

def global_x(xi,nodal_x):
    x=np.zeros(2)
    N=shape_functions(xi)
    for i in range(2):
        for j in range(3):
            x[i]+=nodal_x[i,j]*N[j]
    return x

def jacobian(nodal_x):
    J=np.zeros([2,2])
    for a in range(2): #x or y
        for b in range(2): #xi 1 or 2
            for c in range(3): #x^e 1, 2 or 3
                J[a,b]+=nodal_x[a,c]*div_xi_shape_functions()[b,c]
    return J 

def div_x_shape_functions(x):
    dNxi=div_xi_shape_functions()
    J_inv=np.linalg.inv(jacobian(x))
    return J_inv.T@dNxi

def det_J(J):
    return J[0,0]*J[1,1]-J[0,1]*J[1,0]

def integrate_psi(psi):
    inte_psi=0
    xi=np.array([[1/6,4/6,1/6],[1/6,1/6,4/6]])
    for j in range(3):
        inte_psi+=psi(xi[:,j])/6
    return inte_psi

def integrate_phi(phi,x_nodal):
    xi=np.array([[1/6,4/6,1/6],[1/6,1/6,4/6]])
    detJ=abs(det_J(jacobian(x_nodal)))
    integrand = lambda xi: detJ*phi(global_x(xi,x_nodal))
    return integrate_psi(integrand)

def stiffness_2d(nodes):
    div = div_x_shape_functions(nodes)
    k=np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            phi = lambda x: div[0,i]*div[0,j]+div[1,i]*div[1,j]
            k[i,j] = integrate_phi(phi, nodes)
    return k

def stiffness_advection(nodes):
    div = div_x_shape_functions(nodes)
    k = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            psi=lambda xi: abs(det_J(jacobian(nodes)))*div[1,i]*shape_functions(xi)[j]
            k[i,j] += integrate_psi(psi)

    return k

def force_2d(nodes,S):
    xi=np.array([[1/6,4/6,1/6],[1/6,1/6,4/6]])
    detJ=abs(det_J(jacobian(nodes)))
    f = np.zeros(3)
    for b in range(3):
        integrand = lambda xi: abs(detJ) * S(global_x(xi,nodes)) * shape_functions(xi)[b]
        f[b] = integrate_psi(integrand)
    return f

def generate_2d_grid(Nx):
    Nnodes = Nx+1
    x = np.linspace(0, 1, Nnodes)
    y = np.linspace(0, 1, Nnodes)
    X, Y = np.meshgrid(x,y)
    nodes = np.zeros((Nnodes**2,2))
    nodes[:,0] = X.ravel()
    nodes[:,1] = Y.ravel()
    ID = np.zeros(len(nodes), dtype=np.int64)
    boundaries = dict() # Will hold the boundary values
    n_eq = 0
    for nID in range(len(nodes)):
        if np.allclose(nodes[nID, 0], 0):
            ID[nID] = -1
            boundaries[nID] = 0 # Dirichlet BC
        else:
            ID[nID] = n_eq
            n_eq += 1
            if ( (np.allclose(nodes[nID, 1], 0)) or (np.allclose(nodes[nID, 0], 1)) or (np.allclose(nodes[nID, 1], 1)) ):
                boundaries[nID] = 0 # Neumann BC
    IEN = np.zeros((2*Nx**2, 3), dtype=np.int64)
    for i in range(Nx):
        for j in range(Nx):
            IEN[2*i+2*j*Nx , :] = (i+j*Nnodes,
                                    i+1+j*Nnodes,
                                    i+(j+1)*Nnodes)
            IEN[2*i+1+2*j*Nx, :] = (i+1+j*Nnodes,
                                    i+1+(j+1)*Nnodes,
                                    i+(j+1)*Nnodes)
    return nodes, IEN, ID, boundaries

def source_function(x):
    mean=[442365, 115483] #UoS coords
    std=1000
    return np.exp((-1/(2*std**2))*((.5*(x[0]-mean[0])**2+.5*(x[1]-mean[1])**2)))

def solver(S=source_function,D=1,u=1E-10,map='esw',res='100'):
    #loading data    
    nodes = np.loadtxt('data/'+map+'_nodes_'+res+'k.txt')
    
    IEN = np.loadtxt('data/'+map+'_IEN_'+res+'k.txt', 
                    dtype=np.int64)
    boundary_nodes = np.loadtxt('data/'+map+'_bdry_'+res+'k.txt', 
                                dtype=np.int64)
    southern_boarder = np.where(nodes[boundary_nodes,1] <= 110000)[0]

    ID = np.zeros(len(nodes), dtype=np.int64)
    n_eq = 0
    for nID in range(len(nodes)):
        if nID in southern_boarder:
            ID[nID] = -1
        else:
            ID[nID] = n_eq
            n_eq += 1

    N_equations = np.max(ID)+1
    N_elements = IEN.shape[0]
    N_nodes = nodes.shape[0]
    nodes=nodes.T
    
    # Location matrix
    LM = np.zeros_like(IEN.T)
    for e in range(N_elements):
        for a in range(3):
            LM[a,e] = ID[IEN[e,a]]
    # Global stiffness matrix and force vector
    K = sp.lil_matrix((N_equations, N_equations))
    F = np.zeros((N_equations,))
    # Loop over elements
    for e in range(N_elements):
        k_e = D*stiffness_2d(nodes[:,IEN[e,:]]) - u*stiffness_advection(nodes[:,IEN[e,:]])
        f_e = force_2d(nodes[:,IEN[e,:]], S)
        for a in range(3):
            A = LM[a, e]
            for b in range(3):
                B = LM[b, e]
                if (A >= 0) and (B >= 0):
                    K[A, B] += k_e[a, b]
            if (A >= 0):
                F[A] += f_e[a]
    # Solve
    K=sp.csr_matrix(K)
    Psi_interior = sp.linalg.spsolve(K, F)
    Psi_A = np.zeros(N_nodes)
    for n in range(N_nodes):
        if ID[n] >= 0: # Otherwise Psi should be zero, and we've initialized that already.
            Psi_A[n] = Psi_interior[ID[n]]

    plt.cla()
    plt.tripcolor(nodes[0], nodes[1],Psi_A, triangles=IEN)
    plt.scatter(442365, 115483,c='k',marker='*',label='UoS')
    plt.scatter(473993, 171625,c='k',marker='*', label='UoR')
    plt.title('finite element solver')
    # plt.colorbar()
    plt.axis('equal')
    plt.savefig('test_u_'+str(u)+'_D_'+str(D)+'_'+map+'_'+res+'_.pdf')
    plt.show()

    #plot_solution_and_analytical(IEN, Psi_A, psi_analytical, nodes)

    return Psi_A

def plot_solution_and_analytical(IEN, Psi_A, psi_analytical, nodes):
    z=psi_analytical(nodes)
    vmin = min(np.min(Psi_A), np.min(z))
    vmax = max(np.max(Psi_A), np.max(z))

    fig, ax = plt.subplots(1, 2)
    c1=ax[0].tripcolor(nodes[0], nodes[1],Psi_A, triangles=IEN,vmin=vmin,vmax=vmax)
    ax[0].axis('equal')
    ax[0].set_title('finite element solver')

    c2=ax[1].tripcolor(nodes[0], nodes[1],z, triangles=IEN,vmin=vmin,vmax=vmax)
    ax[1].set_title('analytical solution')

    fig.colorbar(c1, ax=[ax[0], ax[1]])
    plt.show()

solver()
solver(map='las',res='40')
solver(u=0,D=1)
solver(u=1,D=0)