import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse as sp

def shape_functions(xi):
    """Finds the shape functions of the reference triangle.

    Args:
        xi (array_like): list or array of lenght 2. Reference coordinates.

    Returns:
        NumPy array: Shape function at xi
    """
    return np.array([1-xi[0]-xi[1], xi[0], xi[1]])

def div_xi_shape_functions():
    """Finds the derivative of the shape function with respect to reference xi.

    Returns:
        NumPy array: Derivative of the shape function at reference xi.
    """
    return np.array([[-1,1,0],[-1,0,1]])

def global_x(xi,nodal_x):
    """Translates local coordinates and reference xi to the global coordinates.

    Args:
        xi (array_like): list or array of lenght 2. Reference coordinates. 
        nodal_x (array_like): list or array with size (2,3). Local coordinates

    Returns:
        NumPy array: Global coordinates
    """
    x=np.zeros(2)
    N=shape_functions(xi)
    for i in range(2):
        for j in range(3):
            x[i]+=nodal_x[i,j]*N[j]
    return x

def jacobian(nodal_x):
    """Finds the Jacobian at given local coordinates

    Args:
        nodal_x (array_like): list or array with size (2,3). Local coordinates

    Returns:
        NumPy array with size (2,2): The Jacobian at the local coordinates
    """
    J=np.zeros([2,2])
    for a in range(2): #x or y
        for b in range(2): #xi 1 or 2
            for c in range(3): #x^e 1, 2 or 3
                J[a,b]+=nodal_x[a,c]*div_xi_shape_functions()[b,c]
    return J 

def div_x_shape_functions(x):
    """Calculates the derivative of the shape function with respect to local coordicates

    Args:
        x (array_like): list or array with size (2,3). Local coordinates

    Returns:
        NumPy array with size (2,3): derivative of the shape function with respect to local coordicates
    """
    dNxi=div_xi_shape_functions()
    J_inv=np.linalg.inv(jacobian(x))
    return J_inv.T@dNxi

def det_J(J):
    """Finds the determinate of a given (2,2) matrix 

    Args:
        J (array_like): list or array with size (2,2)

    Returns:
        float: the determinate of the (2,2) matrix 
    """
    
    return J[0,0]*J[1,1]-J[0,1]*J[1,0]

def integrate_psi(psi):
    """Finds the integral of a function over an element

    Args:
        psi (function): a function that can be integrated over an element

    Returns:
        float: integral of psi
    """
    inte_psi=0
    xi=np.array([[1/6,4/6,1/6],[1/6,1/6,4/6]])
    for j in range(3):
        inte_psi+=psi(xi[:,j])/6
    return inte_psi

def integrate_phi(phi,x_nodal):
    """Finds the integral of a function over the refrence triangle

    Args:
        phi (function): function defined over the reference triangle
        x_nodal (array_like): list or array with size (2,3). Local coordinates

    Returns:
        float: integral of phi
    """
    
    detJ=abs(det_J(jacobian(x_nodal)))
    integrand = lambda xi: detJ*phi(global_x(xi,x_nodal))
    return integrate_psi(integrand)

def stiffness_diffusion(nodes):
    """finds the diffusion stiffness matrix element

    Args:
        nodes (array_like): list or array with size (2,3). Local coordinates

    Returns:
        NumPy array with size (3,3): A matrix for calculating an element of the diffusion stiffness matrix
    """
    div = div_x_shape_functions(nodes)
    k=np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            phi = lambda x: div[0,i]*div[0,j]+div[1,i]*div[1,j]
            k[i,j] = integrate_phi(phi, nodes)
    return k

def stiffness_advection(nodes):
    """finds the advection stiffness matrix element

    Args:
        nodes (array_like): list or array with size (2,3). Local coordinates

    Returns:
        NumPy array with size (3,3): A matrix for calculating an element of the advection stiffness matrix
    """
    div = div_x_shape_functions(nodes)
    k = np.zeros([3,3])
    detJ=det_J(jacobian(nodes))
    for i in range(3):
        for j in range(3):
            psi=lambda xi: detJ*shape_functions(xi)[i]*div[1,j] + .5*detJ*shape_functions(xi)[i]*div[0,j]
            k[i,j] += integrate_psi(psi)

    return k

def force_2d(nodes,S,t):
    """finds the force vector

    Args:
        nodes (array_like): list or array with size (2,3). Local coordinates
        S (function): source function

    Returns:
        NumPy array with lenght 3: vector for finding the force vector
    """
    detJ=abs(det_J(jacobian(nodes)))
    f = np.zeros(3)
    for b in range(3):
        integrand = lambda xi: abs(detJ) * S(global_x(xi,nodes),t) * shape_functions(xi)[b]
        f[b] = integrate_psi(integrand)
    return f

def mass_matrix(nodes):
    """finds the mass matrix. Used for time dependance, since this was not fully developed, this function is unused

    Args:
        nodes (array_like): list or array with size (2,3). Local coordinates

    Returns:
        NumPy array: The mass matrix at the given nodes
    """
    detJ=det_J(jacobian(nodes))
    M=sp.lil_matrix([3,3])
    print(M)
    for i in range(3):
        for j in range(3):
            integrad=lambda xi: detJ*shape_functions(xi)[i]*shape_functions(xi)[j]
            M[i,j]=integrate_psi(integrad)
    return sp.csr_matrix(M)

def curlyF(nodes,F,K,Psi):
    """Finds the time derivative of the shape function.

    Args:
        nodes (array_like): list or array with size (2,3). Local coordinates
        F (array_like): force vector
        K (array_like): stiffness matrix
        Psi (array_like): concentration of the pollutant

    Returns:
        array_like: the time derivative of the shape function.
    """
    M=mass_matrix(nodes)
    return np.linalg.inv(M)*(F-K*Psi)

def source_function(x,t):
    """Generates Gaussian source function

    Args:
        x (array_like): list or array with lenght 2. Global coordinates

    Returns:
        float: value of Gaussian at x
    """
    mean=[442365, 115483] #UoS coords
    std=1000
    return np.exp((-1/(2*std**2))*((.5*(x[0]-mean[0])**2+.5*(x[1]-mean[1])**2)))

def solver(S=source_function,D=10000,u=10,map='esw',res='100',max_time=1,dt=1,error_calc=False,plotting=False):
    """finds the finite elements solution.

    Args:
        S (function, optional): Source function for the pollutant. Defaults to source_function.
        D (int, optional): Diffusion coefficent. [m^2s^-1] Defaults to 1.
        u (int, optional): Wind speed. [ms^-1] Defaults to 1E-10.
        map (str, optional): Chooses which map to solve over. Choose between 'esw' and 'lan'. Defaults to 'esw'.
        res (str, optional): Chooses resolution to solve over. Defaults to '100'.
            Choose between '6_25', '12_5', '25', '50' or '100' for map = 'ews' or '1_25', '2_5', '5', '10', '20', '40' for map = 'las'.  

    Returns:
        NumPy array: finite elements solution
    """
    Nt=int(max_time/dt)

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
    Psi=np.zeros([Nt,len(nodes[0])])

    fig,ax=plt.subplots()
    plt.triplot(nodes[0],nodes[1],triangles=IEN)
    ax.scatter(442365, 115483,c='k',marker='.',label='UoS',edgecolors='none')
    ax.scatter(473993, 171625,c='r',marker='.', label='UoR',edgecolors='none')
    plt.title(res+'km resolution grid')
    plt.savefig('grid_'+res+'.pdf')

    # Location matrix
    LM = np.zeros_like(IEN.T)
    for e in range(N_elements):
        for a in range(3):
            LM[a,e] = ID[IEN[e,a]]
    K = sp.lil_matrix((N_equations, N_equations))
    for e in range(N_elements):
        k_e = D*stiffness_diffusion(nodes[:,IEN[e,:]]) - u*stiffness_advection(nodes[:,IEN[e,:]])
        for a in range(3):
            A = LM[a, e]
            for b in range(3):
                B = LM[b, e]
                if (A >= 0) and (B >= 0):
                    K[A, B] += k_e[a, b]
    K=sp.csr_matrix(K)

    for t in range(Nt):
        F = np.zeros((N_equations,))
        # Loop over elements
        for e in range(N_elements):
            f_e = force_2d(nodes[:,IEN[e,:]], S,dt*t)
            for a in range(3):
                A = LM[a, e]
                if (A >= 0):
                    F[A] += f_e[a]
        # Solve
        Psi_interior = sp.linalg.spsolve(K, F)
        Psi_A = np.zeros(N_nodes)
        for n in range(N_nodes):
            if ID[n] >= 0: # Otherwise Psi should be zero, and we've initialized that already.
                Psi_A[n] = Psi_interior[ID[n]]

        #normalising
        Psi_A=(Psi_A-min(Psi_A))/(max(Psi_A)-min(Psi_A))

        #saving Psi_A at time t
        Psi[t]=Psi_A

        #potential time dependence implementation
        # F_a=curlyF(nodes,F,K,Psi_A)
        # Psi[t+1]=Psi[t]+dt*F_a

    if plotting:
        fig,ax=plt.subplots()
        pc=ax.tripcolor(nodes[0], nodes[1],Psi_A, triangles=IEN, vmin=Psi_A.min(), vmax=Psi_A.max())
        ax.scatter(442365, 115483,c='k',marker='.',label='UoS',edgecolors='none')
        ax.scatter(473993, 171625,c='r',marker='.', label='UoR',edgecolors='none')
        # ax.scatter(nodes[0,IEN[IEN_tri_index]],nodes[1,IEN[IEN_tri_index]],marker='.',edgecolors='none',s=)
        plt.title('finite element solver')
        cbar = plt.colorbar(pc, ax=ax)
        plt.axis('equal')
        plt.legend(loc='upper right')
        plt.savefig('u_'+str(u)+'_D_'+str(D)+'_'+map+'_'+res+'.pdf')
        plt.show()

    tri=which_triangle(nodes,IEN)
    IEN_tri_index=np.where(np.all(np.sort(IEN,axis=1) == np.sort(tri), axis=1))[0]
    Psi_UoR=Psi[:,IEN[IEN_tri_index]][:,0,:]
    reading=[473993, 171625]
    dists=np.zeros(3)
    dists[0]=((reading[0]-nodes[0,IEN[IEN_tri_index][0][0]])**2+(reading[1]-nodes[1,IEN[IEN_tri_index][0][0]])**2)**.5
    dists[1]=((reading[0]-nodes[0,IEN[IEN_tri_index][0][1]])**2+(reading[1]-nodes[1,IEN[IEN_tri_index][0][1]])**2)**.5
    dists[2]=((reading[0]-nodes[0,IEN[IEN_tri_index][0][2]])**2+(reading[1]-nodes[1,IEN[IEN_tri_index][0][2]])**2)**.5
    final_ans=np.average(Psi_UoR,axis=1,weights=dists)

    if error_calc:
        return final_ans[0], N_equations
    else:
        return final_ans[0]

def triangle_area(p0,p1,p2):
    """finds the area of a triangle made of the three input nodes

    Args:
        p0 (array_like): has lenght 2. first point on the triangle
        p1 (array_like): has lenght 2. first point on the triangle
        p2 (array_like): has lenght 2. first point on the triangle

    Returns:
        float: the area of the triangle
    """
    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = p2
    
    return 0.5 * abs(x1*y2 + x2*y3 + x3*y1 - (y1*x2 + y2*x3 + y3*x1))

def in_tri(tri,node):
    """determines if a point is in a given triangel by checking triangle areas.

    Args:
        tri (array_like): has shape (3,2) The nodes that make up a triangle
        node (array_like): has lenght 2. The node to check if it is in the triangle

    Returns:
        bool: True if the node is in the triangle, False otherwise.
    """
    p0, p1, p2 = tri.T
    
    Area = triangle_area(p0, p1, p2)
    area1 = triangle_area(p0, p1, node)
    area2 = triangle_area(p0, p2, node)
    area3 = triangle_area(p1, p2, node)

    epsilon = 10 # to allow for rounding errors in the calculation
    return abs(Area - (area1 + area2 + area3)) < epsilon

def which_triangle(nodes,IEN):
    """fins the triangle that contains Reading from the grid.

    Args:
        nodes (array_like): has shape (N,2) the nodes of the grid
        IEN (array_like): has shape (M,3) gives the nodes in each triangle

    Returns:
        array_like with lenght: the node index for the triangle containing reading
    """
    N_elements = IEN.shape[0]
    reading=[473993, 171625]
    for e in range(N_elements):
        tf = in_tri((nodes[:,IEN[e,:]]),reading)
        if tf:
            return IEN[e,:]

def l2_error(aim,pred):
        """Finds the l^2 error

        Args:
            aim (array): The target value. In the case of this experiment, it is the analytical solution.
            pred (array): The predicted value. In the case of this experiment, it is the numeric solution.

        Returns:
            l2_error: int
        """
        return (((pred-aim)**2))**.5/((aim**2))**.5

def error(plotting=False):
    """finds and plots the l2 error for the experiment

    Args:
        plotting (bool, optional): If True, plots the solutions to Psi on each grid. Defaults to False.

    Returns:
        array_like: a list of errors for each resolution.
    """
    true_psi=solver(map='las',res='1_25',plotting=plotting)
    reses_str=['2_5', '5', '10', '20', '40']
    E=np.zeros(len(reses_str))
    Ns=[]
    for i,v in enumerate(reses_str):
        psi,N=solver(map='las',res=v,error_calc=True,plotting=plotting)
        E[i]=l2_error(true_psi,psi)
        Ns.append(N)

    fig,ax=plt.subplots()
    Ns=np.array(Ns)
    plt.loglog(Ns,E,'bo',label='$l^2$-error')

    m,c=np.polyfit(np.log(Ns[:]),np.log(E[:]),1)
    x=np.linspace(min(Ns),max(Ns))
    y=np.exp(c)*x**m 

    plt.loglog(x,y,'r-',label='line of best fit, m = '+str(round(m,3)))
    plt.loglog(x,np.exp(c+1.1)*x**-1,'g',label='m = -1')

    plt.legend()
    plt.xlabel('Degrees of freedom')
    plt.ylabel('$l^2$-error')
    plt.grid('both')
    plt.title('Convergence of error')
    plt.savefig('error.pdf')
    return E