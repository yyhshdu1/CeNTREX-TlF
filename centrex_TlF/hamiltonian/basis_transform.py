import numpy as np
from tqdm import tqdm

__all__ = [
    'generate_transform_matrix'
]

def generate_transform_matrix(basis1, basis2, progress = False):
    """
    Function that generates a transform matrix that takes Hamiltonian expressed
    in basis1 to basis2: H_2 = S.conj().T @ H_1 @ S

    inputs:
    basis1 = list of basis states that defines basis1
    basis2 = list of basis states that defines basis2
    progress = boolean to display tqdm progress bar

    returns:
    S = transformation matrix that takes Hamiltonian (or any operator) from 
    basis1 to basis2
    """

    #Check that the two bases have the same dimension
    if len(basis1) != len(basis2):
        print("Bases don't have the same dimension")
        return 1
    
    #Initialize S
    S = np.zeros((len(basis1), len(basis1)), dtype = complex)

    #Loop over the bases and calculate inner products
    for i, state1 in enumerate(tqdm(basis1, disable = not progress)):
        for j in range(len(basis2)):
            state2 = basis2[j]
            val = state1@state2
            S[i,j] = val
            if i != j:
                S[j,i] = np.conjugate(val) 

    #Return the transform matrix
    return S