import functions as f
import numpy as np

def source_function(x):
    
    mean=[442365, 115483] #UoS coords
    std=1000
    return np.exp((-1/(2*std**2))*((.5*(x[0]-mean[0])**2+.5*(x[1]-mean[1])**2)))

psi=f.solver(source_function)
# print(psi)