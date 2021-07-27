import scipy
import numpy as np
import centrex_TlF
from centrex_TlF.hamiltonian.basis_transform import generate_transform_matrix
import centrex_TlF.states as states
from centrex_TlF.hamiltonian.generate_hamiltonian import (
    generate_coupled_hamiltonian_B, generate_uncoupled_hamiltonian_X
)
from centrex_TlF.hamiltonian.utils import (
    matrix_to_states, reorder_evecs, generate_uncoupled_hamiltonian_X_function, 
    matrix_to_states, reduced_basis_hamiltonian, 
    generate_coupled_hamiltonian_B_function
)

__all__ = [
    'generate_reduced_X_hamiltonian', 'generate_reduced_B_hamiltonian',
]

def generate_diagonalized_hamiltonian(hamiltonian, keep_order = True, 
        return_V_ref = False, rtol = None):
    D, V = np.linalg.eigh(hamiltonian)
    if keep_order:
        V_ref = np.eye(V.shape[0])
        D, V = reorder_evecs(V,D,V_ref)

    hamiltonian_diagonalized = V.conj().T @ hamiltonian @ V
    if rtol:
        hamiltonian_diagonalized[
            np.abs(hamiltonian_diagonalized) < 
            np.abs(hamiltonian_diagonalized).max()*rtol] = 0
    if not return_V_ref or not keep_order:
        return hamiltonian_diagonalized, V
    else:
        return hamiltonian_diagonalized, V, V_ref

def generate_reduced_X_hamiltonian(ground_states_approx, Jmin = 0, Jmax = 4, 
                                    E = np.array([0,0,0]),
                                    B = np.array([0,0,0.001]),
                                    rtol = None):

    QN = states.generate_uncoupled_states_ground(list(range(Jmin,Jmax+1)))
    QNc = states.generate_coupled_states_ground(list(range(Jmin,Jmax+1)))
    H_X_uc = generate_uncoupled_hamiltonian_X(QN)
    H_X_uc = generate_uncoupled_hamiltonian_X_function(H_X_uc)
    S_transform = generate_transform_matrix(QN, QNc)

    H_X = S_transform.conj().T @ H_X_uc(E,B) @ S_transform
    if rtol:
        H_X[np.abs(H_X) < np.abs(H_X).max()*rtol] = 0

    # diagonalize the Hamiltonian
    H_X_diag, V, V_ref = generate_diagonalized_hamiltonian(H_X, 
                            keep_order = True, return_V_ref = True, rtol = rtol
        )

    # new set of quantum numbers:
    QN_diag = matrix_to_states(V, QNc)

    ground_states = states.find_exact_states(
                                    [1*gs for gs in ground_states_approx], 
                                    H_X_diag, QN_diag, V_ref = V_ref
        )

    # ground_states = [gs.remove_small_components() for gs in ground_states]

    H_X_red = reduced_basis_hamiltonian(QN_diag, H_X_diag, ground_states)

    return ground_states, H_X_red

def generate_reduced_B_hamiltonian(excited_states_approx, 
                                    E = np.array([0,0,0]),
                                    B = np.array([0,0,0.001]),
                                    rtol = 1e-14):
    for qn in excited_states_approx:
        assert qn.isCoupled, "supply list of CoupledBasisStates"
    H_B =  generate_coupled_hamiltonian_B(excited_states_approx)
    H_B = generate_coupled_hamiltonian_B_function(H_B)

    H_B_diag, V, V_ref_B = generate_diagonalized_hamiltonian(H_B(E, B),
                                                            keep_order = True, 
                                                            return_V_ref = True,
                                                            rtol = rtol)

    # new set of quantum numbers:
    QN_B_diag = matrix_to_states(V, excited_states_approx)

    excited_states = centrex_TlF.states.find_exact_states([1*e for e in excited_states_approx], 
                                            H_B_diag, QN_B_diag, V_ref=V_ref_B)

    H_B_red = reduced_basis_hamiltonian(QN_B_diag, H_B_diag, excited_states)
    return excited_states, H_B_red

def generate_total_hamiltonian(H_X_red, H_B_red, element_limit = 0.1):
    H_X_red[np.abs(H_X_red) < element_limit] = 0
    H_B_red[np.abs(H_B_red) < element_limit] = 0

    H_int = scipy.linalg.block_diag(H_X_red, H_B_red)
    V_ref_int = np.eye(H_int.shape[0])

    return H_int, V_ref_int