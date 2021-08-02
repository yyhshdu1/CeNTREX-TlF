import numpy as np
from centrex_TlF.constants.constants import I_Tl, I_F
from centrex_TlF.states.utils import parity_X
from centrex_TlF.states.states import UncoupledBasisState, CoupledBasisState

__all__ = [
    'generate_uncoupled_states_ground', 
    'generate_uncoupled_states_excited',
    'generate_coupled_states_ground',
    'generate_coupled_states_excited'
]

def generate_uncoupled_states_ground(Js):
    # convert J to int(J); np.int with (-1)**J throws an exception for negative J
    QN = np.array([UncoupledBasisState(int(J),mJ,I_Tl,m1,I_F,m2, Omega = 0, 
                                        P = parity_X(J), 
                                        electronic_state='X')
          for J in Js
          for mJ in range(-J,J+1)
          for m1 in np.arange(-I_Tl,I_Tl+1)
          for m2 in np.arange(-I_F,I_F+1)])
    return QN

def generate_uncoupled_states_excited(Js, Ωs = [-1,1]):
    QN = np.array([UncoupledBasisState(J,mJ,I_Tl,m1,I_F,m2, Omega = Ω,
                                        electronic_state='B')
          for Ω in Ωs
          for J in Js
          for mJ in range(-J,J+1)
          for m1 in np.arange(-I_Tl,I_Tl+1)
          for m2 in np.arange(-I_F,I_F+1)])
    return QN

def generate_coupled_states_ground(Js):
    QN =  np.array([CoupledBasisState(F,mF,F1,J,I_F,I_Tl, 
            electronic_state='X', P = parity_X(J), Omega = 0)
            for J  in Js
            for F1 in np.arange(np.abs(J-I_F),J+I_F+1)
            for F in np.arange(np.abs(F1-I_Tl),F1+I_Tl+1)
            for mF in np.arange(-F, F+1)
            ])
    return QN

def generate_coupled_states_excited(Js, Fs = None, F1s = None, Ps = None):
    if not Fs:
        if not Ps:
            Ps = [+1]
        QN =  np.array([CoupledBasisState(F,mF,F1,J,I_F,I_Tl, 
            electronic_state='B', P = P, Omega = 1)
            for J  in Js
            for F1 in np.arange(np.abs(J-I_F),J+I_F+1)
            for F in np.arange(np.abs(F1-I_Tl),F1+I_Tl+1)
            for mF in np.arange(-F, F+1)
            for P in Ps
            ])
    else:
        assert None not in [Fs, F1s, Ps], \
            "need to supply lists of F, F1 or P if one of them is used as an input parameter"
        QN = np.array([
            CoupledBasisState(F,mF,F1,J,I_F,I_Tl, electronic_state='B', P = P, 
                                Omega = 1)
            for J,F1,F in zip(Js, F1s, Fs)
            for mF in np.arange(-F, F+1)
            for P in Ps
        ])
    return QN
        