import numpy as np

__all__ = [
    "check_transition_coupled_allowed", "assert_transition_coupled_allowed"
]

def check_transition_coupled_allowed(state1, state2, ΔmF_allowed, 
                                    return_err = True):
    """Check whether the transition is allowed based on the quantum numbers

    Args:
        state1 (CoupledBasisState): ground CoupledBasisState
        state2 (CoupledBasisState): excited CoupledBasisState
        return_err (boolean): boolean flag for returning the error message

    Returns:
        tuple: (allowed boolean, error message)
    """
    ΔF = int(state2.F - state1.F)
    ΔmF = int(state2.mF - state1.mF)
    ΔP = int(state2.P - state1.P)
    
    flag_ΔP = np.abs(ΔP) != 2
    flag_ΔF = np.abs(ΔF) > 1
    flag_ΔmF = np.abs(ΔmF) != ΔmF_allowed
    flag_ΔFΔmF = (not flag_ΔmF) & (ΔF == 0 & ΔmF == 0)
    
    errors = ""
    if flag_ΔP:
        errors += f"parity invalid"
    if flag_ΔF:
        if len(errors) != 0:
            errors += ", "
        errors += f"ΔF invalid -> ΔF = {ΔF}"
    if flag_ΔmF:
        if len(errors) != 0:
            errors += ", "
        errors += f"ΔmF invalid -> ΔmF = {ΔmF}"
    
    if flag_ΔFΔmF:
        if len(errors) != 0:
            errors += ", "
        errors += f"ΔF = 0 & ΔmF = 0 invalid"
    
    if len(errors) != 0:
        errors = f"transition not allowed; {errors}" 
    
    if return_err:
        return not (flag_ΔP | flag_ΔF | flag_ΔmF | flag_ΔFΔmF), errors
    else:
        return not (flag_ΔP | flag_ΔF | flag_ΔmF | flag_ΔFΔmF)
    
def assert_transition_coupled_allowed(state1, state2, ΔmF_allowed):
    """Check whether the transition is allowed based on the quantum numbers.
    Raises an AssertionError if the transition is not allowed.

    Args:
        state1 (CoupledBasisState): ground CoupledBasisState
        state2 (CoupledBasisState): excited CoupledBasisState

    Returns:
        tuple: (allowed boolean, error message)
    """
    allowed, errors = check_transition_coupled_allowed(state1, state2, ΔmF_allowed)
    assert allowed, errors
    return allowed