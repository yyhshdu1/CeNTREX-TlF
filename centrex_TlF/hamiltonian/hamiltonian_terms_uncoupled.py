import numpy as np
import centrex_TlF.states as states
from sympy.physics.wigner import wigner_3j
import centrex_TlF.constants.constants_X as cst_X
import centrex_TlF.constants.constants_B  as cst_B
from centrex_TlF.states.states import State, UncoupledBasisState

from centrex_TlF.hamiltonian.quantum_operators import (
    com, J2, I1z, I1p, Jz, Jm, I1m, Jp, I2m, I2p, I1x, I2x, I2y, I1y, Jy, Jx, 
    I2z, J4, J6
)

########################################################
### Rotational Term
########################################################

def Hrot_X(psi):
    return cst_X.B_rot_X * J2(psi)

########################################################
### Terms with angular momentum dot products
########################################################

def Hc1(psi):
    return cst_X.c1 * ( com(I1z,Jz,psi) + (1/2)*(com(I1p,Jm,psi)+com(I1m,Jp,psi)) )

def Hc2(psi):
    return cst_X.c2 * ( com(I2z,Jz,psi) + (1/2)*(com(I2p,Jm,psi)+com(I2m,Jp,psi)) )

def Hc4(psi):
    return cst_X.c4 * ( com(I1z,I2z,psi) + (1/2)*(com(I1p,I2m,psi)+com(I1m,I2p,psi)) )

def Hc3a(psi):
    return 15*cst_X.c3/cst_X.c1/cst_X.c2 * com(Hc1,Hc2,psi) / ((2*psi.J+3)*(2*psi.J-1))

def Hc3b(psi):
    return 15*cst_X.c3/cst_X.c1/cst_X.c2 * com(Hc2,Hc1,psi) / ((2*psi.J+3)*(2*psi.J-1))

def Hc3c(psi):
    return -10*cst_X.c3/cst_X.c4/cst_X.B_rot_X * com(Hc4,Hrot_X,psi) / ((2*psi.J+3)*(2*psi.J-1))

def Hc3(psi):
     return Hc3a(psi) + Hc3b(psi) + Hc3c(psi)

########################################################
### Field free X state Hamiltonian
########################################################

def Hff_X(psi):
    return Hrot_X(psi) + Hc1(psi) + Hc2(psi) + Hc3a(psi) + Hc3b(psi) \
            + Hc3c(psi) + Hc4(psi)

########################################################
### Λ doubling term
########################################################

def H_LD(psi):
    J = psi.J
    mJ = psi.mJ
    I1 = psi.I1
    m1 = psi.m1
    I2 = psi.I2
    m2 = psi.m2
    Omega = psi.Omega
    Omegaprime = -Omega
    
    amp = (cst_B.q*(-1)**(J-Omegaprime)/(2*np.sqrt(6)) * wigner_3j(J,2,J,-Omegaprime,Omegaprime-Omega, Omega)
           *np.sqrt((2*J-1)*2*J*(2*J+1)*(2*J+2)*(2*J+3)) )
    ket = UncoupledBasisState(J, mJ, I1, m1, I2, m2, Omegaprime,
                                electronic_state = psi.electronic_state)
    
    return State([(amp,ket)])

########################################################
### C'(Tl) - term (Brown 1978 "A determination of fundamental Zeeman parameters 
### for the OH radical", eqn A12)
########################################################

def H_c1p(psi):
    #Find the quantum numbers of the input state
    J = psi.J
    mJ = psi.mJ
    I1 = psi.I1
    m1 = psi.m1
    I2 = psi.I2
    m2 = psi.m2
    Omega = psi.Omega
    
    #I1, I2 and m2 must be the same for non-zero matrix element
    I1prime = I1
    m2prime = m2
    I2prime = I2
    
    #To have non-zero matrix element need OmegaPrime = -Omega
    Omegaprime = -Omega
    
    #q is chosen such that q == Omegaprime
    q = Omega
    
    #Initialize container for storing states and matrix elements
    data = []
    
    #Loop over the possible values of quantum numbers for which the matrix element can be non-zero
    #Need Jprime = J+1 ... |J-1|
    for Jprime in range(np.abs(J-1), J+2):    
        #Loop over possible values of mJprime and m1prime
        for mJprime in np.arange(-Jprime,Jprime+1):
            #Must have mJ+m1 = mJprime + m1prime
            m1prime = mJ+m1-mJprime
            if np.abs(m1prime <= I1):
                #Evaluate the matrix element

                #Matrix element for T(J)T(I)
                term1 = ((-1)**(Jprime-Omegaprime+I1-m1-q+mJprime)*np.sqrt(Jprime*(Jprime+1)*(2*Jprime+1)**2*(2*J+1)*I1*(I1+1)
                                                                           *(2*I1+1))
                         * complex(wigner_3j(Jprime,1,J,-mJprime,mJprime-mJ, mJ)) * complex(wigner_3j(I1,1,I1,-m1prime, m1prime-m1, m1))
                         * complex(wigner_3j(Jprime, 1, J, 0, -q, Omega)) * complex(wigner_3j(Jprime, 1, Jprime, -Omegaprime, -q, 0)))

                #Matrix element for T(I)T(J)
                term2 = ((-1)**(mJprime+J-Omegaprime+I1-m1-q)*np.sqrt(J*(J+1)*(2*J+1)**2*(2*Jprime+1)*I1*(I1+1)
                                                                                   *(2*I1+1))
                        *complex(wigner_3j(Jprime,1,J,-mJprime,mJprime-mJ,mJ)) * complex(wigner_3j(Jprime,1,J,-Omegaprime,-q,0))
                        *complex(wigner_3j(J,1,J,0,-q,Omega)) * complex(wigner_3j(I1,1,I1,-m1prime, m1prime-m1, m1)))

                amp = cst_B.c_Tl*0.5*(term1+term2)

                basis_state = UncoupledBasisState(Jprime, mJprime, I1prime, m1prime, I2prime, m2prime, Omegaprime,
                                P = psi.P, 
                                electronic_state = psi.electronic_state)

                if amp != 0:
                        data.append((amp, basis_state))
                    
    return State(data)

########################################################
### Electron magnetic hyperfine operator
########################################################

def H_mhf_Tl(psi):
    #Find the quantum numbers of the input state
    J = psi.J
    mJ = psi.mJ
    I1 = psi.I1
    m1 = psi.m1
    I2 = psi.I2
    m2 = psi.m2
    Omega = psi.Omega
    
    #I1, I2 and m2 must be the same for non-zero matrix element
    I2prime = I2
    m2prime = m2
    I1prime = I1
    
    #Initialize container for storing states and matrix elements
    data = []
    
    #Loop over the possible values of quantum numbers for which the matrix element can be non-zero
    #Need Jprime = J+1 ... |J-1|
    for Jprime in np.arange(np.abs(J-1), J+2):
        #Evaluate the part of the matrix element that is common for all p
        common_coefficient = cst_B.h1_Tl*wigner_3j(J, 1, Jprime, -Omega, 0, Omega)*np.sqrt((2*J+1)*(2*Jprime+1)*I1*(I1+1)*(2*I1+1))
        
        #Loop over the spherical tensor components of I1:
        for p in np.arange(-1,2):
            #To have non-zero matrix element need mJ-p = mJprime
            mJprime = mJ + p
            
            #Also need m2 - p = m2prime
            m1prime = m1 - p
            
            #Check that mJprime and m2prime are physical
            if np.abs(mJprime) <= Jprime and np.abs(m1prime) <= I1prime:
                #Calculate rest of matrix element
                p_factor = ((-1)**(p-mJ+I1-m1-Omega)*wigner_3j(J, 1, Jprime, -mJ, -p, mJprime)
                               *wigner_3j(I1, 1, I1prime, -m1, p, m1prime))
                               
                amp = Omega*common_coefficient*p_factor
                basis_state = UncoupledBasisState(Jprime, mJprime, I1prime, m1prime, I2prime, m2prime, psi.Omega)
                if amp != 0:
                    data.append((amp, basis_state))
    
    return State(data)

def H_mhf_F(psi):
    #Find the quantum numbers of the input state
    J = psi.J
    mJ = psi.mJ
    I1 = psi.I1
    m1 = psi.m1
    I2 = psi.I2
    m2 = psi.m2
    Omega = psi.Omega
    
    #I1, I2 and m1 must be the same for non-zero matrix element
    I1prime = I1
    m1prime = m1
    I2prime = I2
    
    #Initialize container for storing states and matrix elements
    data = []
    
    #Loop over the possible values of quantum numbers for which the matrix element can be non-zero
    #Need Jprime = J+1 ... |J-1|
    for Jprime in np.arange(np.abs(J-1), J+2):
        #Evaluate the part of the matrix element that is common for all p
        common_coefficient = cst_B.h1_F*wigner_3j(J, 1, Jprime, -Omega, 0, Omega)*np.sqrt((2*J+1)*(2*Jprime+1)*I2*(I2+1)*(2*I2+1))
        
        #Loop over the spherical tensor components of I2:
        for p in np.arange(-1,2):
            #To have non-zero matrix element need mJ-p = mJprime
            mJprime = mJ + p
            
            #Also need m2 - p = m2prime
            m2prime = m2 - p
            
            #Check that mJprime and m2prime are physical
            if np.abs(mJprime) <= Jprime and np.abs(m2prime) <= I2prime:
                #Calculate rest of matrix element
                p_factor = ((-1)**(p-mJ+I2-m2-Omega)*wigner_3j(J, 1, Jprime, -mJ, -p, mJprime)
                               *wigner_3j(I2, 1, I2prime, -m2, p, m2prime))
                               
                amp = Omega*common_coefficient*p_factor
                basis_state = UncoupledBasisState(Jprime, mJprime, I1prime, m1prime, I2prime, m2prime, psi.Omega)
                if amp != 0:
                    data.append((amp, basis_state))
        
        
    return State(data)

########################################################
### Field free B state Hamiltonian
########################################################

# def Hff_B(psi):
#     return Hrot_B(psi) + H_mhf_Tl(psi) + H_mhf_F(psi) + H_c1p(psi) + H_LD(psi)

########################################################
### Zeeman X state
########################################################

def HZx_X(psi):
    if psi.J != 0:
        return -cst_X.μ_J/psi.J*Jx(psi) - cst_X.μ_Tl/psi.I1*I1x(psi) - cst_X.μ_F/psi.I2*I2x(psi)
    else:
        return -cst_X.μ_Tl/psi.I1*I1x(psi) - cst_X.μ_F/psi.I2*I2x(psi)

def HZy_X(psi):
    if psi.J != 0:
        return -cst_X.μ_J/psi.J*Jy(psi) - cst_X.μ_Tl/psi.I1*I1y(psi) - cst_X.μ_F/psi.I2*I2y(psi)
    else:
        return -cst_X.μ_Tl/psi.I1*I1y(psi) - cst_X.μ_F/psi.I2*I2y(psi)
    
def HZz_X(psi):
    if psi.J != 0:
        return -cst_X.μ_J/psi.J*Jz(psi) - cst_X.μ_Tl/psi.I1*I1z(psi) - cst_X.μ_F/psi.I2*I2z(psi)
    else:
        return -cst_X.μ_Tl/psi.I1*I1z(psi) - cst_X.μ_F/psi.I2*I2z(psi)

########################################################
### Zeeman B state Hamiltonian
########################################################

def HZx_B(psi):
    # TODO
    return psi

def HZy_B(psi):
    # TODO
    return psi

def HZz_B(psi):
    #Find the quantum numbers of the input state
    J = psi.J
    mJ = psi.mJ
    I1 = psi.I1
    m1 = psi.m1
    I2 = psi.I2
    m2 = psi.m2
    Omega = psi.Omega
    S = 1
    
    #The other state must have the same value for I1,m1,I2,m2,mJ and Omega
    I1prime = I1
    m1prime = m1
    I2prime = I2
    m2prime = m2
    Omegaprime = Omega
    mJprime = mJ
    
    #Initialize container for storing states and matrix elements
    data = []
    
    #Loop over possible values of Jprime
    for Jprime in range(np.abs(J-1),J+2):
        
        #Electron orbital angular momentum term
        L_term = (cst_B.gL * Omega *np.sqrt((2*J+1)*(2*Jprime+1)) * (-1)**(mJprime-Omegaprime)
                  * complex(wigner_3j(Jprime,1,J,-mJprime,0,mJ)) * complex(wigner_3j(Jprime,1,J,-Omegaprime,0,Omega)))
        
        #Electron spin term
        S_term = (cst_B.gS * np.sqrt((2*J+1)*(2*Jprime+1)) * (-1)**(mJprime-Omegaprime)
                  * complex(wigner_3j(Jprime,1,J,-mJprime,0,mJ)) * complex(wigner_3j(Jprime,1,J,-Omegaprime,0,Omega))
                  * (-1)**(S)*complex(wigner_3j(S,1,S,0,0,0)) * np.sqrt(S*(S+1)*(2*S+1)))
        
        amp = L_term+S_term
        basis_state = UncoupledBasisState(Jprime, mJprime, I1prime, m1prime, I2prime, m2prime, Omegaprime)
        
        
        if amp != 0:
            data.append((amp, basis_state))
                  
                  
    return State(data)

########################################################
### Stark Hamiltonian
########################################################

def HSx(psi):
    return -cst_X.D_TlF * ( R1m(psi) - R1p(psi) ) / np.sqrt(2)

def HSy(psi):
    return -cst_X.D_TlF * 1j * ( R1m(psi) + R1p(psi) ) / np.sqrt(2)

def HSz(psi):
    return -cst_X.D_TlF*R10(psi)

# Old functions from Jakobs original Hamiltonian

def R10(psi):
    amp1 = np.sqrt(2)*np.sqrt((psi.J-psi.mJ)*(psi.J+psi.mJ)/(8*psi.J**2-2))
    ket1 = UncoupledBasisState(psi.J-1, psi.mJ, psi.I1, psi.m1, psi.I2, psi.m2,
                                Omega = psi.Omega, P = states.parity_X(psi.J-1), 
                                electronic_state = psi.electronic_state)
    amp2 = np.sqrt(2)*np.sqrt((psi.J-psi.mJ+1)*(psi.J+psi.mJ+1)/(6+8*psi.J*(psi.J+2)))
    ket2 = UncoupledBasisState(psi.J+1, psi.mJ, psi.I1, psi.m1, psi.I2, psi.m2,
                                Omega = psi.Omega, P = states.parity_X(psi.J+1), 
                                electronic_state = psi.electronic_state)
    return State([(amp1,ket1),(amp2,ket2)])

def R1m(psi):
    amp1 = -.5*np.sqrt(2)*np.sqrt((psi.J+psi.mJ)*(psi.J+psi.mJ-1)/(4*psi.J**2-1))
    ket1 = UncoupledBasisState(psi.J-1, psi.mJ-1, psi.I1, psi.m1, psi.I2, psi.m2,
                                Omega = psi.Omega, P = states.parity_X(psi.J-1), 
                                electronic_state = psi.electronic_state)
    amp2 = .5*np.sqrt(2)*np.sqrt((psi.J-psi.mJ+1)*(psi.J-psi.mJ+2)/(3+4*psi.J*(psi.J+2)))
    ket2 = UncoupledBasisState(psi.J+1, psi.mJ-1, psi.I1, psi.m1, psi.I2, psi.m2,
                                Omega = psi.Omega, P = states.parity_X(psi.J+1), 
                                electronic_state = psi.electronic_state)
    return State([(amp1,ket1),(amp2,ket2)])

def R1p(psi):
    amp1 = -.5*np.sqrt(2)*np.sqrt((psi.J-psi.mJ)*(psi.J-psi.mJ-1)/(4*psi.J**2-1))
    ket1 = UncoupledBasisState(psi.J-1, psi.mJ+1, psi.I1, psi.m1, psi.I2, psi.m2,
                                Omega = psi.Omega, P = states.parity_X(psi.J-1), 
                                electronic_state = psi.electronic_state)
    amp2 = .5*np.sqrt(2)*np.sqrt((psi.J+psi.mJ+1)*(psi.J+psi.mJ+2)/(3+4*psi.J*(psi.J+2)))
    ket2 = UncoupledBasisState(psi.J+1, psi.mJ+1, psi.I1, psi.m1, psi.I2, psi.m2,
                                Omega = psi.Omega, P = states.parity_X(psi.J+1), 
                                electronic_state = psi.electronic_state)
    return State([(amp1,ket1),(amp2,ket2)])

def HI1R(psi):
    return com(I1z,R10,psi) + (com(I1p,R1m,psi)-com(I1m,R1p,psi))/np.sqrt(2)

def HI2R(psi):
    return com(I2z,R10,psi) + (com(I2p,R1m,psi)-com(I2m,R1p,psi))/np.sqrt(2)

def Hc3_alt(psi):
    return 5*cst_X.c3/cst_X.c4*Hc4(psi) - 15*cst_X.c3/2*(com(HI1R,HI2R,psi)+com(HI2R,HI1R,psi))

def Hff_X_alt(psi):
    return Hrot_X(psi) + Hc1(psi) + Hc2(psi) + Hc3_alt(psi) + Hc4(psi)