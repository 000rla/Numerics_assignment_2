import functions as f

def source_function(x):
    # return 2*x[0]*(x[0]-2)*(3*x[1]**2-3*x[1]+.5)+x[1]**2*(x[1]-1)**2
    return 1

def psi_analytical(x):
    # return x[0]*(1-x[0]/2)+x[1]**2*(1-x[1])**2
    return x[0]*(1-x[0]/2)

f.solver(source_function)