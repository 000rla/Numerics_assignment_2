import functions as f
import numpy as np
import matplotlib.pyplot as plt

def main():
    reses=['1_25','2_5', '5', '10', '20', '40']
    psi=np.zeros(len(reses))
    for i,v in enumerate(reses):
        psi[i]=f.solver(map='las',res=v)
    print(psi)
    # fig,ax=plt.subplots()
    # plt.scatter(range(len(reses)),psi)
    # plt.savefig('psis.pdf')
    # plt.show()

main()