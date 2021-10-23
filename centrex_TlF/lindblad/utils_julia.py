import numpy as np
import sympy as smp
from julia import Main
from pathlib import Path
from sympy import Symbol
from dataclasses import dataclass
from collections import OrderedDict
from typing import Union, SupportsFloat

NumberType = type(SupportsFloat)


__all__ = [
    'initialize_julia', 'generate_ode_fun_julia', 'setup_variables_julia',
    'odeParameters', 'setup_parameter_scan_1D', 'setup_ratio_calculation',
    'setup_initial_condition_scan', 'setup_state_integral_calculation',
    'get_indices_diag_flattened', 'setup_parameter_scan_ND', 
    "handle_randomized_ensemble_solution", "generate_preamble"
]

def initialize_julia(nprocs):
    Main.eval("""
        using Logging: global_logger
        using TerminalLoggers: TerminalLogger
        global_logger(TerminalLogger())

        using Distributed
    """)

    if Main.eval("nprocs()") < nprocs:
        Main.eval(f"addprocs({nprocs}-nprocs())")

    if Main.eval("nprocs()") > nprocs:
        procs = Main.eval("procs()")
        procs = procs[nprocs:]
        Main.eval(f"rmprocs({procs})")

    Main.eval("""
        @everywhere begin
            using LinearAlgebra
            using Trapz
            using DifferentialEquations
        end
    """)
    # loading common julia functions from julia_common.jl
    path = Path(__file__).parent / "julia_common.jl"
    Main.eval(f'include(raw"{path}")')

    print(f"Initialized Julia with {nprocs} processes")

def generate_ode_fun_julia(preamble, code_lines):
    ode_fun = preamble
    for cline in code_lines:
        ode_fun += "\t\t"+cline+'\n'
    ode_fun += '\t end \n \t nothing \n end'
    Main.eval(f"@everywhere {ode_fun}")
    return ode_fun

def setup_variables_julia(Γ, ρ, vars = None):
    Main.Γ = Γ
    Main.ρ = ρ
    Main.eval("""
        @everywhere begin
            Γ = $Γ
            ρ = $ρ
        end
    """)
    if vars:
        for key, val in vars.items():
            Main.eval(f"@everywhere {key} = {val}")

@dataclass
class OdeParameter:
    var: str
    val: Union[NumberType, str]

class odeParameters:
    def __init__(self, *args, **kwargs):
        # if elif statement is for legacy support, where a list of parameters was supplied
        if len(args) > 1:
            raise AssertionError("For legacy support supply a list of strings, one for each parameter")
        elif len(args) == 1:
            assert isinstance(args[0][0], str), "For legacy support supply a list of strings, one for each parameter"
            kwargs = {par: 0 for par in args[0]}
            odeParameters(**kwargs)
        
        self._parameters = [key for key,val in kwargs.items() if not isinstance(val, str)]
        self._compound_vars = [key for key,val in kwargs.items() if isinstance(val, str)]
        
        for key,val in kwargs.items():
            # OdeParameter(ϕ = 'ϕ') results in different unicode representation
            # replace the rhs with the rep. used on the lhs
            if isinstance(val, str):
                val = val.replace("\u03d5", "\u03c6")
            setattr(self, key, val)
        
        self._order_compound_vars()
        self.check_symbols_defined()
    
    
    def __setattr__(self, name, value):
        if name in ['_parameters', '_compound_vars']:
            super(odeParameters, self).__setattr__(name, value)
        elif name in self._parameters or self._compound_vars:
            super(odeParameters, self).__setattr__(name, OdeParameter(name, value))
        else:
            raise AssertionError("Cannot instantiate new parameter on initialized OdeParameters object")
    
    def _get_defined_symbols(self):
        symbols_defined = [par for par,val in self.__dict__.items() if isinstance(val, OdeParameter)]
        symbols_defined += ['t']
        symbols_defined = set([Symbol(s) for s in symbols_defined])
        return symbols_defined
    
    def _get_numerical_symbols(self):
        symbols_numerical = [par for par,val in self.__dict__.items() if isinstance(val, OdeParameter)
                            if not isinstance(val.val, str)]
        symbols_numerical += ['t']
        symbols_numerical = set([Symbol(s) for s in symbols_numerical])
        return symbols_numerical
    
    def _get_expression_symbols(self):
        symbols_expressions = [smp.parsing.sympy_parser.parse_expr(val.val) for val in self.__dict__.values()
                               if (isinstance(val, OdeParameter) and isinstance(val.val, str))]
        symbols_expressions = set().union(*[s.free_symbols for s in symbols_expressions])
        return symbols_expressions
    
    def check_symbols_defined(self):
        symbols_defined = self._get_defined_symbols()
        symbols_expressions = self._get_expression_symbols()
        
        warn_flag = False
        warn_string = f"Symbol(s) not defined: "
        for se in symbols_expressions:
            if se not in symbols_defined:
                warn_flag = True
                warn_string += f"{se}, "
        if warn_flag:
            raise AssertionError(warn_string.strip(' ,'))
            
    def _order_compound_vars(self):
        symbols_num = self._get_numerical_symbols()
        unordered = list(self._compound_vars)
        ordered = []

        while len(unordered) != 0:
            for compound in unordered:
                if compound not in ordered:
                    symbols = smp.parsing.sympy_parser.parse_expr(getattr(self, compound).val).free_symbols
                    m = [True if (s in symbols_num) or (str(s) in ordered) else False for s in symbols]
                    if all(m):
                        ordered.append(compound)
            unordered = [val for val in unordered if val not in ordered]
        self._compound_vars = ordered
            
    def _get_index_parameter(self, parameter, mode = 'python'):
        # OdeParameter(ϕ = 'ϕ') results in different unicode representation
        # replace the rhs with the rep. used on the lhs
        parameter = parameter.replace("\u03d5", "\u03c6")
        if mode == 'python':
            return self._parameters.index(parameter)
        elif mode == 'julia':
            return self._parameters.index(parameter)+1
    
    @property
    def p(self):
        return [getattr(self, p).val for p in self._parameters]
        
        
    def get_index_parameter(self, parameter, mode = 'python'):
        if isinstance(parameter, str):
            return self._get_index_parameter(parameter, mode)
        elif isinstance(parameter, (list, np.ndarray)):
            return [self._get_index_parameter(par, mode) for par in parameter]
        
    def check_transition_symbols(self, transitions):
        to_check = ('Ω', 'δ')
        symbols_defined = [str(s) for s in self._get_defined_symbols()]
        not_defined = []
        for transition in transitions:
            for var in to_check:
                var = str(getattr(transition,var))
                if var is not None: 
                    if var not in symbols_defined:
                        not_defined.append(var)
        if len(not_defined) > 0:
            not_defined = set([str(s) for s in not_defined])
            raise AssertionError(f"Symbol(s) from transitions not defined: {', '.join(not_defined)}")
    
    def generate_p_julia(self):
        Main.eval(f"p = {self.p}")
    
    def __repr__(self):
        rep = f"OdeParameters("
        for par in self._parameters:
            rep += f"{par}={getattr(self, par).val}, "
        return rep.strip(", ") + ")"

def setup_parameter_scan_1D(odePar, parameter, values):
    if isinstance(parameter, (list, tuple)):
        indices = [odePar.get_index_parameter(par) for par in parameter]
    else:
        indices = [odePar.get_index_parameter(parameter)]

    pars = str(odePar.generate_p()).strip('[]').split(',')
    for idx in indices:
        pars[idx] = "params[i]"
    
    pars = "[" + ",".join(pars) + "]"
    
    Main.params = values
    Main.eval(f"""
    @everywhere params = $params
    @everywhere function prob_func(prob,i,repeat)
        remake(prob, p = {pars})
    end
    """)

def setup_parameter_scan_ND(odePar, parameters, values, randomize = False):
    pars = str(odePar.generate_p()).strip('[]').split(',')

    for idN, parameter in enumerate(parameters):
        if isinstance(parameter, (list, tuple)):
            indices = [odePar.get_index_parameter(par) for par in parameter]
        else:
            indices = [odePar.get_index_parameter(parameter)]
        for idx in indices:
            pars[idx] = f"params[i,{idN+1}]"
    pars = "[" + ",".join(pars) + "]"
    params = np.array(np.meshgrid(*values)).T.reshape(-1,len(values))
    Main.params = params
    if randomize:
        ind_random = np.random.permutation(len(params))
        Main.params = params[ind_random]

    Main.eval(f"""
    @everywhere params = $params
    @everywhere function prob_func(prob, i, repeat)
        remake(prob, p = {pars})
    end
    """)
    if randomize:
        return ind_random

def generate_preamble(odepars: odeParameters, transitions: list) -> str:
    # check if the symbols in transitions are defined by odepars
    odepars.check_transition_symbols(transitions)
    preamble = """function Lindblad_rhs!(du, ρ, p, t)
    \t@inbounds begin
    """
    for idp, par in enumerate(odepars._parameters):
        preamble += f"\t\t{par} = p[{idp+1}]\n"
    for par in odepars._compound_vars:
        preamble += f"\t\t{par} = {getattr(odepars, par).val}\n"
        
    for transition in transitions:
        preamble += f"\t\t{transition.Ω}ᶜ = conj({transition.Ω})\n"
        
    # remove duplicate lines (if multiple transitions have the same Rabi rate symbol or detuning
    preamble = "\n".join(list(OrderedDict.fromkeys(preamble.split("\n"))))
    return preamble

def setup_ratio_calculation(states):
    cmd = ""
    if isinstance(states[0], (list, np.ndarray, tuple)):
        for state in states:
            cmd += f"sum(real(diag(sol.u[end])[{state}]))/sum(real(diag(sol.u[1])[{state}])), "
        cmd = cmd.strip(', ')
        cmd = "[" + cmd + "]"
    else:
        cmd = f"sum(real(diag(sol.u[end])[{states}]))/sum(real(diag(sol.u[1])[{states}]))"

    Main.eval(f"""
    @everywhere function output_func(sol,i)
        if size(sol.u)[1] == 1
            return NaN, false
        else
            val = {cmd}
            return val, false
        end
    end""")

def handle_randomized_ensemble_solution(ind_random, sol_name = 'sim'):
    order_restored = np.argsort(ind_random)
    # Main.params = Main.params[order_restored]
    return np.asarray(Main.params[order_restored]), np.array(Main.eval(f"{sol_name}.u"))[order_restored]

def setup_initial_condition_scan(values):
    Main.params = values
    Main.eval("@everywhere params = $params")
    Main.eval("""
    @everywhere function prob_func(prob,i,repeat)
        remake(prob,u0=params[i])
    end
    """)

def setup_state_integral_calculation(states, nphotons = False, Γ = None):
    """Setup an integration output_function for an EnsembleProblem. 
    Uses trapezoidal integration to integrate the states.
    
    Args:
        states (list): list of state indices to integrate
        nphotons (bool, optional): flag to calculate the number of photons, 
                                    e.g. normalize with Γ
        Γ (float, optional): decay rate in 2π Hz (rad/s), not necessary if already
                                loaded into Julia globals
    """
    if nphotons & Main.eval("@isdefined Γ"):
        Main.eval(f"""
        @everywhere function output_func(sol,i)
            return Γ.*trapz(sol.t, [real(sum(diag(sol.u[j])[{states}])) for j in 1:size(sol)[3]]), false
        end""")
    else:
        if nphotons:
            assert Γ is not None, "Γ not defined as a global in Julia and not supplied to function"
            Main.eval(f"""
            @everywhere function output_func(sol,i)
                return {Γ}.*trapz(sol.t, [real(sum(diag(sol.u[j])[{states}])) for j in 1:size(sol)[3]]), false
            end""")
        else:
            Main.eval(f"""
            @everywhere function output_func(sol,i)
                return trapz(sol.t, [real(sum(diag(sol.u[j])[{states}])) for j in 1:size(sol)[3]]), false
            end""")

def get_indices_diag_flattened(n):
    return np.diag(np.arange(0,n*n).reshape(-1,n))