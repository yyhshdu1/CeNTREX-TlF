from julia import Main
from centrex_TlF.couplings.collapse import collapse_matrices
from centrex_TlF.hamiltonian.generate_reduced_hamiltonian import (
    generate_total_reduced_hamiltonian
)
from centrex_TlF.couplings.coupling_matrix import (
    generate_coupling_field_automatic
)
from centrex_TlF.lindblad.generate_hamiltonian import (
    generate_total_symbolic_hamiltonian
)
from centrex_TlF.lindblad.generate_system_of_equations import (
    generate_system_of_equations_symbolic
)
from centrex_TlF.lindblad.generate_julia_code import (
    system_of_equations_to_lines, generate_preamble
)
from centrex_TlF.states.generate_states import (
    generate_coupled_states_ground_X, generate_coupled_states_excited_B
)
from centrex_TlF.lindblad.utils_julia import (
    initialize_julia, generate_ode_fun_julia
)

__all__ = [
    'generate_OBE_system', 'setup_OBE_system_julia'
]


def generate_OBE_system(system_parameters, transitions, full_output = False):
    ground_states, excited_states, QN, H_int, V_ref_int = \
        generate_total_reduced_hamiltonian(
            ground_states_approx  = generate_coupled_states_ground_X(system_parameters.ground),
            excited_states_approx = generate_coupled_states_excited_B(system_parameters.excited)
                                        )
    couplings = [
        generate_coupling_field_automatic(
            transition.ground,
            transition.excited,
            H_int, QN, V_ref_int, 
            pol_vec = transition.polarizations,
            nprocs = system_parameters.nprocs,)
        for transition in transitions
    ]

    H_symbolic = generate_total_symbolic_hamiltonian(
                                            QN, H_int, couplings, transitions
                                            )

    C_array = collapse_matrices(QN, ground_states, excited_states, gamma = system_parameters.Γ)
    system = generate_system_of_equations_symbolic(
                            H_symbolic, C_array, progress = False, fast = True
                        )
    code_lines = system_of_equations_to_lines(system, nprocs = system_parameters.nprocs)
    if not full_output:
        return QN, couplings, code_lines
    else:
        return QN, couplings, H_symbolic, H_int, C_array, system, code_lines

def setup_OBE_system_julia(system_parameters, ode_parameters, transitions, full_output = False):
    if not full_output:
        QN, couplings, code_lines = generate_OBE_system(system_parameters, transitions)
    else:
        QN, couplings, H_symbolic, H_int, C_array, system, code_lines = \
            generate_OBE_system(system_parameters, transitions, full_output= True)
    preamble = generate_preamble(ode_parameters, transitions)

    initialize_julia(nprocs = system_parameters.nprocs)

    generate_ode_fun_julia(preamble, code_lines)
    Main.eval(f"Γ = {system_parameters.Γ}")
    ode_parameters.generate_p_julia()
    if not full_output:
        return QN
    else:
        return QN, couplings, H_symbolic, H_int, C_array, system, code_lines