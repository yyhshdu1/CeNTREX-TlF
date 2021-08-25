import numpy as np
import scipy.constants as cst

__all__ = [
    "generate_1D_multipass", "generate_2D_multipass", 
    "calculate_power_from_rabi_gaussian_beam",
    "calculate_rabi_from_power_gaussian_beam", 
    "calculate_power_from_rabi_gaussian_beam_microwave",
    "calculate_rabi_from_power_gaussian_beam_microwave"
]

def gaussian(x, μ, σ):
    return np.exp(-(x-μ)**2 / (2 * σ**2))

def gaussian_2d(y,z,μy,μz,σy,σz):
    a = (y-μy)**2/(2*σy**2)
    b = (z-μz)**2/(2*σz**2)
    return np.exp(-(a+b))

def gaussian_amp(x, a, μ, σ):
    return a*np.exp(-(x-μ)**2 / (2 * σ**2))

def gaussian_2d_amp(a,y,z,μy,μz,σy,σz):
    return a*gaussian_2d(y,z,μy,μz,σy,σz)

def multipass_prism_order(passes):
    npass = [1]
    for p in range(1,passes):
        if p%2 == 0:
            npass.append(p+1)
        else:
            npass.append(passes-p)
    return npass

def generate_1D_multipass(x, npasses, loss, σ, spacing):
    reflections = (np.array(multipass_prism_order(npasses))-1)
    amplitudes = np.array([(1-loss)**r for r in reflections])
    beam_locs = np.array([i*spacing for i in range(npasses)])
    multipass =  gaussian_amp(
        x[:,np.newaxis], amplitudes[np.newaxis, :], beam_locs[np.newaxis, :], σ
    ).sum(axis = 1)
    return multipass

def generate_2D_multipass(X, Y, npasses, loss, σx, σy, spacing):
    reflections = (np.array(multipass_prism_order(npasses))-1)
    amplitudes = np.array([(1-loss)**r for r in reflections])
    beam_locs = np.array([i*spacing for i in range(npasses)])
    multipass = gaussian_2d_amp(amplitudes[np.newaxis,:], 
                                      X[:,:,np.newaxis], Y[:,:,np.newaxis], 
                                      0,  beam_locs[np.newaxis, :], 
                                      σx, σy).sum(axis = 2)
    return multipass

def calculate_power_from_rabi_gaussian_beam(
                                Ω, main_coupling, σx, σy, D = 2.6675506e-30
                                ):
    """calculate the required power for a given Ω, given a main transition 
    matrix element

    Args:
        Ω (float): Rabi rate [2π⋅Hz]
        main_coupling (complex): main transition matrix element
        σx (float): σx of beam
        σy (float): σy of beam
        D (float, optional): effective dipole moment for transitions. 
                            Defaults to 2.6675506e-30 for optical TlF.

    Returns:
        float: power [W]
    """
    # Electric field
    E = Ω*cst.hbar/(main_coupling * D)

    # convert to peak intensity
    I = 1/2 * cst.c *cst.epsilon_0 * E**2

    # convert power to amplitude of the gaussian
    P = I* (2*np.pi*σx*σy)

    return P

def calculate_rabi_from_power_gaussian_beam(
                                P, main_coupling, σx, σy, D = 2.6675506e-30,
                                Γ = None):
    """calculate Ω for a given power, given a main transition matrix element

    Args:
        P (float): optical power [W]
        main_coupling (complex): main transition matrix element
        σx (float): σx of beam
        σy (float): σy of beam
        D ([type], optional): effective dipole moment for transitions. 
                                Defaults to 2.6675506e-30 for optical TlF.
        Γ (float, optional): Γ to normalize Rabi rate with
    Returns:
        float: power [W]
    """
    # intensity from power
    I = P / (2*np.pi*σx*σy)

    # electric field from intensity
    E = np.sqrt(I*2/(cst.c*cst.epsilon_0))

    # rabi rate from electric field
    Ω = (E*main_coupling*D)/cst.hbar

    # normalize with Γ if provided
    if Γ:
        Ω /= Γ

    return Ω

def calculate_power_from_rabi_gaussian_beam_microwave(
                                    Ω, main_coupling, σx, σy, D = 1.4103753e-29
                                    ):
    return calculate_power_from_rabi_gaussian_beam(Ω, main_coupling, σx, σy, D)

def calculate_rabi_from_power_gaussian_beam_microwave(
                                    P, main_coupling, σx, σy, D = 1.4103753e-29,
                                    Γ = None
                                    ):
    return calculate_rabi_from_power_gaussian_beam(P, main_coupling, σx, σy, D, Γ)