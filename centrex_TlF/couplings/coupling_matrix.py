import sqlite3
import logging
import numpy as np
import multiprocessing
from pathlib import Path
from centrex_TlF.couplings.utils_sqlite import (
    check_states_in_ED_ME_coupled
)
from centrex_TlF.couplings.matrix_elements import (
    generate_ED_ME_mixed_state, calculate_ED_ME_mixed_state
)
from centrex_TlF.couplings.utils_multiprocessing import multi_coupling_matrix
from centrex_TlF.states.utils import (
    find_exact_states, check_approx_state_exact_state
)

__all__ = [
    'calculate_coupling_matrix', 'generate_coupling_field', 'generate_D',
    'generate_coupling_matrix', 'calculate_coupling_field'
]

def generate_coupling_matrix(QN, ground_states, excited_states, 
                            pol_vec = np.array([0,0,1]), reduced = False,
                            nprocs = 1):
    """generate optical coupling matrix for given ground and excited states

    Args:
        QN (list): list of basis states
        ground_states (list): list of ground states coupling to excited states
        excited_states (list): list of excited states
        pol_vec (np.ndarray, optional): polarization vector. Defaults to np.array([0,0,1]).
        reduced (bool, optional): [description]. Defaults to False.
        nrpocs (int): # processes to use for multiprocessing

    Returns:
        np.ndarray: optical coupling matrix
    """
    assert isinstance(QN, list), "QN required to be of type list"

    # check if states are pre-cached
    Jg = np.unique([s[1].J for state in QN for s in state if s[1].electronic_state == "X"])
    Je = np.unique([s[1].J for state in QN for s in state if s[1].electronic_state == "B"])
    pre_cached = check_states_in_ED_ME_coupled(Jg, Je, pol_vec)

    if pre_cached:
        if nprocs > 1:
            logging.warning("generate_coupling_matrix: Pre-cached calculations, multiprocessing not used")

        # connect to sqlite3 database on file, not used when multiprocessing
        path = Path(__file__).parent.parent / "pre_calculated"
        db = path / "matrix_elements.db"

        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute("PRAGMA synchronous = OFF")
        cur.execute("PRAGMA journal_mode = MEMORY")
        # initialize the coupling matrix
        H = np.zeros((len(QN),len(QN)), dtype = complex)

        # start looping over ground and excited states
        for ground_state in ground_states:
            i = QN.index(ground_state)
            for excited_state in excited_states:
                j = QN.index(excited_state)

                # calculate matrix element and add it to the Hamiltonian
                H[i,j] = generate_ED_ME_mixed_state(
                                            ground_state, 
                                            excited_state, 
                                            pol_vec = pol_vec, 
                                            reduced = reduced,
                                            con = con)
        con.close()
        H = H + H.conj().T
        return H
    else:
        return calculate_ED_ME_mixed_state(QN, ground_states, excited_states,
                                            pol_vec, reduced, nprocs)

def calculate_coupling_matrix(QN, ground_states, excited_states, 
                            pol_vec = np.array([0,0,1]), reduced = False,
                            nprocs = 1):
    """generate optical coupling matrix for given ground and excited states

    Args:
        QN (list): list of basis states
        ground_states (list): list of ground states coupling to excited states
        excited_states (list): list of excited states
        pol_vec (np.ndarray, optional): polarization vector. Defaults to np.array([0,0,1]).
        reduced (bool, optional): [description]. Defaults to False.
        nrpocs (int): # processes to use for multiprocessing

    Returns:
        np.ndarray: optical coupling matrix
    """
    assert isinstance(QN, list), "QN required to be of type list"

    if nprocs > 1:
        with multiprocessing.Pool(nprocs) as pool:
            result = pool.starmap(multi_coupling_matrix,
                [(QN, gs, excited_states, pol_vec, reduced) 
                    for gs in ground_states])
        H = np.sum(result, axis = 0)
    else:
        # initialize the coupling matrix
        H = np.zeros((len(QN),len(QN)), dtype = complex)

        # start looping over ground and excited states
        for ground_state in ground_states:
            i = QN.index(ground_state)
            for excited_state in excited_states:
                j = QN.index(excited_state)

                # calculate matrix element and add it to the Hamiltonian
                H[i,j] = calculate_ED_ME_mixed_state(
                                            ground_state, 
                                            excited_state, 
                                            pol_vec = pol_vec, 
                                            reduced = reduced
                                            )

    # make H hermitian
    H = H + H.conj().T

    return H

def generate_D(H, QN, ground_main, excited_main, excited_states, Δ = 0):
    # find transition frequency
    ig = QN.index(ground_main)
    ie = QN.index(excited_main)
    ω0 = (H[ie,ie] - H[ig,ig]).real

    # calculate the shift Δ = ω - ω₀
    ω = ω0 + Δ

    # shift matrix
    D = np.zeros(H.shape, H.dtype)
    for excited_state in excited_states:
        idx = QN.index(excited_state)
        D[idx,idx] -= ω

    return D

def generate_coupling_field(ground_main_approx, excited_main_approx, 
                            ground_states_approx, excited_states_approx, 
                            H_rot, QN, V_ref, pol_main = np.array([0,0,1]), 
                            pol_vec = [],
                            relative_coupling = 1e-3,
                            absolute_coupling = 1e-6,
                            nprocs = 2):
    ground_states = find_exact_states(
                        ground_states_approx, H_rot, QN, V_ref = V_ref
                                    )
    excited_states = find_exact_states(
                        excited_states_approx, H_rot, QN, V_ref = V_ref
                                    )
    ground_main = find_exact_states([ground_main_approx], 
                                        H_rot, QN, V_ref = V_ref)[0]
    excited_main = find_exact_states([excited_main_approx], 
                                        H_rot, QN, V_ref = V_ref)[0]

    check_approx_state_exact_state(ground_main_approx, ground_main)
    check_approx_state_exact_state(excited_main_approx, excited_main)
    ME_main = generate_ED_ME_mixed_state(
                        excited_main, ground_main, pol_vec = pol_main)

    assert_msg = f"main coupling element small, {ME_main:.2e}" + \
                  ", check states and/or polarization"
    assert np.abs(ME_main) > 1e-2, assert_msg

    couplings = {
        'ground main': ground_main,
        'excited main': excited_main,
        'main coupling': ME_main,
        'ground_states': ground_states,
        'excited states': excited_states,
        'D': generate_D(H_rot, QN, ground_main, excited_main, excited_states),
        'fields': []
    }
    
    for pol in pol_vec:
        coupling = generate_coupling_matrix(
                                            QN, 
                                            ground_states, 
                                            excited_states, 
                                            pol_vec = pol, 
                                            reduced = False,
                                            nprocs = nprocs)

        coupling[np.abs(coupling) < relative_coupling*np.max(np.abs(coupling))] = 0
        coupling[np.abs(coupling) < absolute_coupling] = 0
        d = {'pol': pol, 'field': coupling}
        couplings['fields'].append(d)
    return couplings

def calculate_coupling_field(ground_main_approx, excited_main_approx, 
                            ground_states_approx, excited_states_approx, 
                            H_rot, QN, V_ref, pol_main = np.array([0,0,1]), 
                            pol_vec = [],
                            relative_coupling = 1e-3,
                            absolute_coupling = 1e-6,
                            nprocs = 2):
    ground_states = find_exact_states(
                        ground_states_approx, H_rot, QN, V_ref = V_ref
                                    )
    excited_states = find_exact_states(
                        excited_states_approx, H_rot, QN, V_ref = V_ref
                                    )
    ground_main = find_exact_states([ground_main_approx], 
                                        H_rot, QN, V_ref = V_ref)[0]
    excited_main = find_exact_states([excited_main_approx], 
                                        H_rot, QN, V_ref = V_ref)[0]

    check_approx_state_exact_state(ground_main_approx, ground_main)
    check_approx_state_exact_state(excited_main_approx, excited_main)
    ME_main = calculate_ED_ME_mixed_state(
                        excited_main, ground_main, pol_vec = pol_main)

    assert_msg = f"main coupling element small, {ME_main:.2e}" + \
                  ", check states and/or polarization"
    assert np.abs(ME_main) > 1e-2, assert_msg

    couplings = {
        'ground main': ground_main,
        'excited main': excited_main,
        'main coupling': ME_main,
        'ground_states': ground_states,
        'excited states': excited_states,
        'D': generate_D(H_rot, QN, ground_main, excited_main, excited_states),
        'fields': []
    }
    
    for pol in pol_vec:
        coupling = calculate_coupling_matrix(
                                            QN, 
                                            ground_states, 
                                            excited_states, 
                                            pol_vec = pol, 
                                            reduced = False,
                                            nprocs = nprocs)

        coupling[np.abs(coupling) < relative_coupling*np.max(np.abs(coupling))] = 0
        coupling[np.abs(coupling) < absolute_coupling] = 0
        d = {'pol': pol, 'field': coupling}
        couplings['fields'].append(d)

    return couplings