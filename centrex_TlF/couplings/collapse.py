import numpy as np
from tqdm import tqdm
from centrex_TlF.couplings.branching import calculate_BR

__all__ = [
    'collapse_matrices'
]
def collapse_matrices(QN, ground_states, excited_states, gamma = 1, tol = 1e-4,
                        progress = False):
    """
    Function that generates the collapse matrix for given ground and excited states

    inputs:
    QN = list of states that defines the basis for the calculation
    ground_states = list of ground states that are coupled to the excited states
    excited_states = list of excited states that are coupled to the ground states
    gamma = decay rate of excited states
    tol = couplings smaller than tol/sqrt(gamma) are set to zero to speed up computation

    outputs:
    C_list = list of collapse matrices
    """
    #Initialize list of collapse matrices
    C_list = []

    #Start looping over ground and excited states
    for excited_state in tqdm(excited_states, disable = not progress):
        j = QN.index(excited_state)
        BRs = calculate_BR(excited_state, ground_states)
        if np.sum(BRs) > 1:
            print(f"Warning: Branching ratio sum > 1, difference = {np.sum(BRs)-1:.2e}")
        for ground_state, BR in zip(ground_states, BRs):
            i = QN.index(ground_state)

            if np.sqrt(BR) > tol:
                #Initialize the coupling matrix
                H = np.zeros((len(QN),len(QN)), dtype = complex)
                H[i,j] = np.sqrt(BR*gamma)

                C_list.append(H)

    return C_list