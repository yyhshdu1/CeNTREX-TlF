import numpy as np
from tqdm import tqdm
from centrex_TlF.hamiltonian.hamiltonian_terms_uncoupled import (
    Hff_X, HSx, HSy, HSz, HZx_X, HZy_X, HZz_X, Hff_X_alt
)
from centrex_TlF.hamiltonian.hamiltonian_B_terms_coupled import (
    Hrot_B, H_mhf_Tl, H_mhf_F, H_LD, H_cp1_Tl, H_c_Tl, HZz_B
)

__all__ = [
    'generate_uncoupled_hamiltonian_X', 'generate_coupled_hamiltonian_B'
]

def HMatElems(H, QN, progress = False):
    result = np.zeros((len(QN), len(QN)), dtype=complex)
    for i,a in tqdm(enumerate(QN), total=len(QN), disable = not progress):
        for j in range(i,len(QN)):
            b = QN[j]
            val = (1*a)@H(b)
            result[i,j] = val
            if i != j:
                result[j,i] = np.conjugate(val)
    return result

def generate_uncoupled_hamiltonian_X(QN):
    for qn in QN:
        assert qn.isUncoupled, "supply list with UncoupledBasisStates"
    return {
            "Hff" :  HMatElems(Hff_X_alt, QN),
            "HSx" :  HMatElems(HSx, QN),
            "HSy" :  HMatElems(HSy, QN),
            "HSz" :  HMatElems(HSz, QN),
            "HZx" :  HMatElems(HZx_X, QN),
            "HZy" :  HMatElems(HZy_X, QN),
            "HZz" :  HMatElems(HZz_X, QN),
    }

def generate_coupled_hamiltonian_B(QN):
    for qn in QN:
        assert qn.isCoupled, "supply list withCoupledBasisStates"
    return {
            "Hrot" : HMatElems(Hrot_B, QN),
            "H_mhf_Tl" : HMatElems(H_mhf_Tl, QN),
            "H_mhf_F" : HMatElems(H_mhf_F, QN),
            "H_LD" : HMatElems(H_LD, QN),
            "H_cp1_Tl" : HMatElems(H_cp1_Tl, QN),
            "H_c_Tl" : HMatElems(H_c_Tl, QN),
            "HZz" : HMatElems(HZz_B, QN),
    }