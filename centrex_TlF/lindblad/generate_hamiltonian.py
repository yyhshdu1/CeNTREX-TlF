import copy
import logging
import numpy as np
from sympy import zeros, Symbol
from centrex_TlF.couplings import (
    generate_total_hamiltonian
)
from centrex_TlF.states.utils_compact import (
    compact_QN_coupled_indices, find_indices_to_compact_coupled
)
from centrex_TlF.lindblad.utils_compact import (
    compact_symbolic_hamiltonian_indices, delete_row_column_symbolic
)
__all__ = [
    'generate_symbolic_hamiltonian', 'generate_symbolic_detunings', \
    'generate_total_symbolic_hamiltonian',
]

def generate_symbolic_hamiltonian(QN, H_rot, couplings, Ωs = None,  Δs = None,
                                    pols = None):

    n_states = H_rot.shape[0]

    if not Ωs:
        Ωs = [Symbol(f'Ω{idx}', complex = True) for idx in range(len(couplings))]
    if len(Ωs) != len(couplings):
        Ωs = [Symbol(f'Ω{idx}', complex = True) for idx in range(len(couplings))]
        logging.warning("Warning in generate_symbolic_hamiltonian: supplied " +
                    f"Ωs length does not match # couplings ({len(Ωs)} != {len(couplings)})"
            )
    if not Δs:
        Δs = [Symbol(f'Δ{idx}') for idx in range(len(couplings))]
    if len(Δs) != len(couplings):
        Δs = [Symbol(f'Δ{idx}') for idx in range(len(couplings))]
        logging.warning("Warning in generate_symbolic_hamiltonian: supplied " +
                    f"Δs length does not match # couplings ({len(Δs)} != {len(couplings)})"
            )

    # initialize empty Hamiltonian
    hamiltonian = zeros(*H_rot.shape)

    # add the couplings to the fields 
    for idc, (Ω, coupling) in enumerate(zip(Ωs, couplings)):
        # check if Ω symbol exists, else create
        if not Ω:
            _ = idc
            while True:
                Ω = Symbol(f'Ω{_}', complex = True)
                _ += 1
                if Ω not in Ωs:
                    break
            Ωs[idc] = Ω
        main_coupling = coupling['main coupling']
        for field in coupling['fields']:
            if pols:
                P = pols[idc]
                if P:
                    P = P[tuple(field['pol'])]
                    hamiltonian += (P*Ω/main_coupling)/2 * field['field']
                else:
                    hamiltonian += (Ω/main_coupling)/2 * field['field'] 
            else:
                hamiltonian += (Ω/main_coupling)/2 * field['field']

    # add detunings to the hamiltonian
    for idc, (Δ, coupling) in enumerate(zip(Δs, couplings)):
        # check if Δ symbol exists, else create
        if not Δ:
            _ = idc
            while True:
                Δ = Symbol(f'Δ{_}')
                _ += 1
                if Δ not in Δs:
                    break
            Δs[idc] = Δ
        indices = [QN.index(s) for s in coupling['excited states']]
        for idx in indices:
            hamiltonian[idx,idx] += Δ

    # ensure hermitian Hamiltonian for complex Ω
    # complex conjugate Rabi rates
    Ωsᶜ = [Symbol(str(Ω)+"ᶜ", complex = True) for Ω in Ωs]
    for idx in range(n_states):
        for idy in range(0,idx):
            for Ω,Ωᶜ in zip(Ωs, Ωsᶜ):
                hamiltonian[idx,idy] = hamiltonian[idx,idy].subs(Ω, Ωᶜ)
    
    hamiltonian += H_rot

    return hamiltonian#, symbols

def generate_symbolic_detunings(n_states, detunings):
    detuning = zeros(n_states, n_states)
    
    if len(detunings) == 1:
        for idd, indices in enumerate(detunings):
            Δ = Symbol(f'Δ', real = True)
            for idx in indices:
                detuning[idx, idx] += Δ

        symbols = [Symbol(f'Δ', complex = True)]
    else:
        for idd, indices in enumerate(detunings):
            Δ = Symbol(f'Δ{idd+1}', real = True)
            for idx in indices:
                detuning[idx, idx] += Δ

        symbols = [Symbol(f'Δ{idd+1}', complex = True)
                    for idd in range(len(detunings))]
    return detuning, symbols

def generate_total_symbolic_hamiltonian(QN, H_int, couplings, transitions,
                                        slice_compact = None, qn_compact = None):
    """Generate the total symbolic hamiltonian for the given system

    Args:
        QN (list): states
        H_int (array): internal hamiltonian
        couplings (list): list of dictionaries with all couplings of the system
        transitions (list): list of dictionaries with all transitions of the 
                            system
        slice_compact (slice operator, optional): numpy slice operator for which 
                                                    part of the system to compact 
                                                    to a single state. 
                                                    Defaults to None.
        qn_compact (list, optional): list of dicts with each dict containing the 
                                        quantum numbers to compact into a single 
                                        state. Defaults to None.

    Returns:
        sympy matrix: symbolic hamiltonian
        if qn_compact is provided, also returns the states corresponding to the 
        compacted hamiltonian, i.e. ham, QN_compact
    """
    H_rot = generate_total_hamiltonian(H_int, QN, couplings)
    Ωs = [t.get('Ω symbol') for t in transitions]
    Δs = [t.get('Δ symbol') for t in transitions]
    pols = []
    for transition in transitions:
        if not transition.get('polarization symbols'):
            pols.append(None)
        else:
            _ = {tuple(p): ps for p, ps in  zip(transition['polarizations'], 
                                        transition['polarization symbols'])}
            pols.append(_)

    H_symbolic = generate_symbolic_hamiltonian(QN, H_rot, couplings, Ωs, Δs, 
                                                                        pols)

    if slice_compact:
        H_symbolic = delete_row_column_symbolic(H_symbolic, slice_compact)
    elif qn_compact:
        QN_compact = copy.deepcopy(QN)
        for qnc in qn_compact:
            indices_compact = find_indices_to_compact_coupled(qnc, QN_compact)
            QN_compact = compact_QN_coupled_indices(QN_compact, indices_compact)
            H_symbolic = compact_symbolic_hamiltonian_indices(H_symbolic, indices_compact)
        return H_symbolic, QN_compact

    return H_symbolic