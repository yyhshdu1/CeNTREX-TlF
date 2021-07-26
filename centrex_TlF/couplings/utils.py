from scipy.sparse import kron, eye
from sympy.physics.wigner import wigner_3j, wigner_6j

def threej_f(j1,j2,j3,m1,m2,m3):
    return complex(wigner_3j(j1,j2,j3,m1,m2,m3))

def sixj_f(j1,j2,j3,j4,j5,j6):
    return complex(wigner_6j(j1,j2,j3,j4,j5,j6))

def generate_sharp_superoperator(M, identity = None):
    """
    Given an operator M in Hilbert space, generates sharp superoperator M_L in 
    Liouville space (see "Optically pumped atoms" by Happer, Jau and Walker)
    sharp = post-multiplies density matrix: |rho@A) = A_sharp @ |rho) 

    inputs:
    M = matrix representation of operator in Hilbert space

    outputs:
    M_L = representation of M in in Liouville space
    """

    if identity == None:
         identity = eye(M.shape[0], format = 'coo')

    M_L = kron(M.T,identity, format = 'csr')

    return M_L

def generate_flat_superoperator(M, identity = None):
    """
    Given an operator M in Hilbert space, generates flat superoperator M_L in 
    Liouville space (see "Optically pumped atoms" by Happer, Jau and Walker)
    flat = pre-multiplies density matrix: |A@rho) = A_flat @ |rho)

    inputs:
    M = matrix representation of operator in Hilbert space

    outputs:
    M_L = representation of M in in Liouville space
    """
    if identity == None:
         identity = eye(M.shape[0], format = 'coo')

    M_L = kron(identity, M, format = 'csr')

    return M_L

def generate_superoperator(A,B):
    """
    Function that generates superoperator representing 
    |A@rho@B) = np.kron(B.T @ A) @ |rho)

    inputs:
    A,B = matrix representations of operators in Hilbert space

    outpus:
    M_L = representation of A@rho@B in Liouville space
    """

    M_L = kron(B.T, A, format = 'csr')

    return M_L