import functions as f
import numpy as np

def main(error=False, plotting=True):
    """runs the experiment to find the pollutant over Reading

    Args:
        error (bool, optional): If True, finds the l2-error for each resolution and the order of convergance. Otherwise, finds the value of Psi. Defaults to False.
        plotting (bool, optional): If True, saves the plots of the solutions. Defaults to True.

    Returns:
        array: If error = True, returns the l2-error. Otherwise returns the values of Psi over Reading.
    """
    if error:
        return f.error(plotting=plotting)
    else:
        reses=['1_25','2_5', '5', '10', '20', '40']
        psi=np.zeros(len(reses))
        for i,v in enumerate(reses):
            psi[i]=f.solver(map='las', res=v, plotting=plotting)
        return psi

main(error=True, plotting=True)