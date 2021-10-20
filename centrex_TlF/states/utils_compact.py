import numpy as np

__all__ = [
    ''
]

def find_indices_to_compact_coupled(qn_compact, QN):
    """get the indices corresponding to the quantum numbers to compact into a
    single state for simulation purposes; e.g. for a J level which the excited
    state only decays into

    Args:
        qn_compact (dict): dict containing the quantum numbers to compact into a 
                            single state. J and the electronic state are 
                            required, F1 and F are optional
        QN (list): list of states in the simulation; required to be in the same 
                    order as the hamiltonian

    Returns:
        [list]: list of indices corresponding to each dict in qns
    """
    Js = np.array([s.find_largest_component().J for s in QN])
    F1s = np.array([s.find_largest_component().F1 for s in QN])
    Fs = np.array([s.find_largest_component().F for s in QN])
    estates = np.array([s.find_largest_component().electronic_state for s in QN])
    
    mask_all = np.ones(len(QN), dtype = bool)

    # get the quantum numbers for each part to compact
    J = qn_compact.get('J')
    F1 = qn_compact.get('F1')
    F = qn_compact.get('F')
    estate = qn_compact.get("electronic state")
    assert estate is not None, "supply the electronic state to compact"
    # generate the masks for states in QN where the conditions are met
    mask_J = Js == J if J is not None else mask_all
    mask_F1 = F1s == F1 if F1 is not None else mask_all
    mask_F = Fs == F if F is not None else mask_all
    mask_es = estates == estate if estate is not None else np.zeros(len(QN), dtype == bool)
    # get the indices of the states in QN to compact
    return np.where(mask_J & mask_F1 & mask_F & mask_es)[0]


def compact_QN_coupled_indices(QN, indices_compact):
    """Compact the states given by indices in indices_compact

    Args:
        QN (list): states
        indices_compact (list, array): indices to compact into a single state

    Returns:
        list: compacted states
    """
    QNc = [QN[idx] for idx in indices_compact]

    slc = lambda s: s.find_largest_component()

    Js = np.unique([slc(s).J for s in QNc if slc(s).J is not None])
    F1s = np.unique([slc(s).F1 for s in QNc if slc(s).F1 is not None])
    Fs = np.unique([slc(s).F for s in QNc if slc(s).F is not None])
    mFs = np.unique([slc(s).mF for s in QNc if slc(s).mF is not None])
    Ps = np.unique([slc(s).P for s in QNc if slc(s).P is not None])

    QNcompact = [qn for idx, qn in enumerate(QN) if idx not in indices_compact[1:]]

    state_rep = QNcompact[indices_compact[0]].find_largest_component()
    if len(Js) != 1:
        state_rep.J = None
    if len(F1s) != 1:
        state_rep.F1 = None
    if len(Fs) != 1:
        state_rep.F = None
    if len(mFs) != 1:
        state_rep.mF = None
    if len(Ps) != 1:
        state_rep.P = None

    # make it a state again instead of uncoupled basisstate
    QNcompact[indices_compact[0]] = (1.0 + 0j) * state_rep    

    return QNcompact