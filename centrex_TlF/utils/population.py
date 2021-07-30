import numpy as np
import scipy.constants as cst

__all__ = [
    'thermal_population', 'J_levels', 'J_slice', 'generate_thermal_J'
]

def thermal_population(J, T, B=6.66733e9, n = 100):
    """calculate the thermal population of a given J sublevel

    Args:
        J (int): rotational level
        T (float): temperature [Kelvin]
        B (float, optional): rotational constant. Defaults to 6.66733e9.
        n (int, optional): number of rotational levels to normalize with. 
                            Defaults to 100.

    Returns:
        float: relative population in a rotational sublevel
    """
    c = 2*np.pi*cst.hbar*B/(cst.k*T)
    g = lambda J: 4*(2*J+1)
    a = lambda J: -c*J*(J+1)
    Z = np.sum([g(i)*np.exp(a(i)) for i in range(n)])
    return g(J)*np.exp(a(J))/Z

def J_levels(J):
    """calculate the number of hyperfine sublevels per J rotational level

    Args:
        J (int): rotational level

    Returns:
        int: number of levels
    """
    return 4*(2*J + 1)

def J_slice(J):
    """generate a slice object for a rotational sublevel

    Args:
        J (int): rotational level

    Returns:
        numpy slice: numpy slice object
    """
    if J == 0:
        return np.s_[0:J_levels(0)]
    else:
        levels = J_levels(np.arange(J+1))
        return np.s_[np.sum(levels[:-1]):np.sum(levels)]

def generate_thermal_J(Js, n_excited, T, normalized = True):
    """generate a thermal distribution over the rotational states

    Args:
        Js (list,array): included J levels
        T (int, float): Temperature in Kelvin

    Returns:
        np.ndarray: density matrix
    """
    # calculate the total number of levels in the system
    levels = np.sum([J_levels(J) for J in Js])

    # initialize empty density matrix
    ρ = np.zeros([levels+n_excited, levels+n_excited], dtype = complex)

    index = 0
    for J in Js:
        p = thermal_population(J, T)
        l = J_levels(J)
        sl = np.s_[index:index+l]
        np.fill_diagonal(ρ[sl, sl], p/levels)
        index += l

    if normalized:
        # normalize the density matrix trace to 1
        ρ /= np.sum(np.diag(ρ))
    return ρ