import copy
from sympy import zeros, Symbol

__all__ = [
    'generate_density_matrix_symbolic', 'subs_rabi_rate'
]

def recursive_subscript(i):
    # chr(0x2080+i) is unicode for
    # subscript num(i), resulting in x₀₀ for example
    if i < 10:
        return chr(0x2080+i)
    else:
        return recursive_subscript(i//10)+chr(0x2080+i%10)

def generate_density_matrix_symbolic(levels):
    ρ = zeros(levels,levels)
    levels = levels
    for i in range(levels):
        for j in range(i,levels):
            # \u03C1 is unicode for ρ, 
            if i == j:
                ρ[i,j] = Symbol(u'\u03C1{0},{1}'. \
                format(recursive_subscript(i), recursive_subscript(j)))
            else:
                ρ[i,j] = Symbol(u'\u03C1{0},{1}'. \
                format(recursive_subscript(i), recursive_subscript(j)))
                ρ[j,i] = Symbol(u'\u03C1{1},{0}'. \
                format(recursive_subscript(i), recursive_subscript(j)))
    return ρ

def subs_rabi_rate(hamiltonian, originals, replacement):
    ham = copy(hamiltonian)
    Ωr = Symbol(f'Ω{replacement}', complex = True)
    Ωrᶜ = Symbol(f'Ω{replacement}ᶜ', complex = True)
    for original in originals:
        Ω = Symbol(f'Ω{original}', complex = True)
        Ωᶜ = Symbol(f'Ω{original}ᶜ', complex = True)
        ham = ham.subs((Ω, Ωr), (Ωᶜ, Ωrᶜ))
    return ham