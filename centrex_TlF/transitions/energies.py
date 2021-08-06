import json
import scipy
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from centrex_TlF.states.states import CoupledBasisState
from centrex_TlF.states import (
    BasisStates_from_State,
    find_state_idx_from_state
)
from centrex_TlF.hamiltonian import (
    generate_uncoupled_hamiltonian_X,
    generate_uncoupled_hamiltonian_X_function,
    generate_coupled_hamiltonian_B,
    generate_coupled_hamiltonian_B_function,
    generate_diagonalized_hamiltonian, 
    matrix_to_states
)
from functools import lru_cache

__all__ = [
    'calculate_energies', 'calculate_transition_frequency', 
    'generate_transition_frequency'
]

def calculate_energies(ground_states, excited_states, E = np.array([0,0,0]), 
                    B = np.array([0,0,0]),
                    offset_X = 1*CoupledBasisState(J=0, F1 = 1/2, F = 1, mF = 0, 
                        I1 =1/2, I2 = 1/2, electronic_state='X', P = +1, Omega = 0),
                    offset_B = 1*CoupledBasisState(J=1, F1 = 3/2, F = 2, mF = 0, I1 =1/2, 
                            I2 = 1/2, electronic_state='B', P = -1, Omega = 1),
                    offset = 2*np.pi*1103407960000000.0,
                    nprocs = 1):

    # transform coupled basis to uncoupled
    QN_X_uc = [s.transform_to_uncoupled() for s in ground_states]
    # grab unique BasisStates from the State objects
    QN_X_uc = BasisStates_from_State(QN_X_uc)
    # calculate and diagonalize the X state Hamiltonian
    H_X = generate_uncoupled_hamiltonian_X(QN_X_uc, nprocs = nprocs)
    H_X = generate_uncoupled_hamiltonian_X_function(H_X)(E, B)
    H_X, V_X = generate_diagonalized_hamiltonian(H_X, keep_order = False)
    # generate the corresponding ground states in coupled basis
    QN_X = matrix_to_states(V_X, QN_X_uc)
    QN_X = [s.transform_to_coupled() for s in QN_X]

    # calculate and diagonalize the B state Hamiltonian
    H_B = generate_coupled_hamiltonian_B(excited_states, nprocs = nprocs)
    H_B = generate_coupled_hamiltonian_B_function(H_B)(E, B)
    H_B, V_B = generate_diagonalized_hamiltonian(H_B, keep_order = False)
    QN_B = matrix_to_states(V_B, excited_states)

    # add offset
    idX = find_state_idx_from_state(H_X, offset_X, QN_X)
    idB = find_state_idx_from_state(H_B, offset_B, QN_B)
    Eshift = (H_X[idX, idX] -  H_B[idB,idB]) + offset
    H_B += Eshift*np.eye(len(excited_states))

    H_tot = scipy.linalg.block_diag(H_X, H_B)
    QN = list(QN_X) + list(QN_B)

    return QN, H_tot

def calculate_transition_frequency(state1, state2, H, QN):
    id1 = find_state_idx_from_state(H, state1, QN)
    id2 = find_state_idx_from_state(H, state2, QN)

    E1 = H[id1,id1].real
    E2 = H[id2,id2].real

    return E2-E1

def _check_precached(state, config = None):
    if not config:
        path = Path(__file__).parent.parent / "pre_calculated"
        js = path / "precalculated.json"
        with open(js) as json_file:
            config = json.load(json_file)

    Js = np.unique([s.J for _,s in state])
    es = state.find_largest_component().electronic_state
    for J in Js:
        # if B state check if J + 1 is included in pre-cached values to account
        # for mixing
        if es == 'B':
            J += 1
        assert J in config['transitions'][es], f'Hamiltonian term for |{es}, {J}> not pre-cached'

@lru_cache(maxsize = 2**16)
def generate_transition_frequency(state1, state2):
    """Get the field freetransition frequency between state1 and state2 in 2π⋅Hz.
    Grabs pre-cached version, otherwise throws exception.

    Args:
        state1 (State): TlF State
        state2 (State): TlF State

    Returns:
        frequency: transition frequency in 2π⋅Hz
    """
    path = Path(__file__).parent.parent / "pre_calculated"
    js = path / "precalculated.json"

    # check if state1 and state2 are included in the pre-cached transitions
    _check_precached(state1)
    _check_precached(state2)
    
    
    with open(path / 'transitions.pickle', 'rb') as f:
        _ = pickle.load(f)
        QN, H = _['QN'], _['H']

    return calculate_transition_frequency(state1, state2, H, QN)
    
    