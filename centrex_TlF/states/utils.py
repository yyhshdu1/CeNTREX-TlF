import numpy as np
from centrex_TlF.hamiltonian.utils import reorder_evecs

__all__ = [
    'find_state_idx_from_state', 'find_exact_states', 
    'check_approx_state_exact_state', 'parity_X'
]

def parity_X(J):
    return (-1)**J

def find_state_idx_from_state(H, reference_state, QN, V_ref = None):
    """Determine the index of the state vector most closely corresponding to an
    input state

    Args:
        H (np.ndarray): Hamiltonian to compare to
        reference_state (State): state to find closest state in H to
        QN (list): list of state objects defining the basis for H

    Returns:
        int: index of closest state vector of H corresponding to reference_state
    """
    
    # determine state vector of reference state
    reference_state_vec = reference_state.state_vector(QN)
    
    # find eigenvectors of the given Hamiltonian
    E, V = np.linalg.eigh(H)
    
    if V_ref is not None:
        E, V = reorder_evecs(V,E,V_ref)
    
    overlaps = np.dot(np.conj(reference_state_vec),V)
    probabilities = overlaps*np.conj(overlaps)
    
    idx = np.argmax(probabilities)
    
    return idx

def find_exact_states(states_approx, H, QN, V_ref = None):
    """Find closest approximate eigenstates corresponding to states_approx

    Args:
        states_approx (list): list of State objects
        H (np.ndarray): Hamiltonian, diagonal in basis QN
        QN (list): list of State objects defining the basis for H

    Returns:
        list: list of eigenstates of H closest to states_approx
    """
    states = []
    for state_approx in states_approx:
        i = find_state_idx_from_state(H, state_approx, QN, V_ref)
        states.append(QN[i])

    return states

def check_approx_state_exact_state(approx, exact):
    approx = approx.find_largest_component()
    exact = exact.find_largest_component()

    assert approx.electronic_state == exact.electronic_state, \
        f"mismatch in electronic state {approx.electronic_state} != {exact.electronic_state}"
    assert approx.J == exact.J, \
        f"mismatch in J {approx.J} != {exact.J}"
    assert approx.F == exact.F, \
        f"mismatch in F {approx.F} != {exact.F}"
    assert approx.F1 == exact.F1, \
        f"mismatch in F1 {approx.F1} != {exact.F1}"
    assert approx.mF == exact.mF, \
        f"mismatch in mF {approx.mF} != {exact.mF}"