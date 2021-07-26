# CeNTREX-TlF
 
Collection of code tools written for the CeNTREX TlF Schiff moment experiment.

Consists of three main components:
* `states`
* `hamiltonians`
* `couplings`

# Installation
To install run `python setup.py install`, which will install the `centrex_TlF` package.

Dependencies are:
* tqdm
* scipy
* numpy
* sympy
* multiprocessing

# Description

## `states`
`states` contains the functions and classes to represent the TlF states:  
`CoupledBasisState` is a class representing a TlF state with coupled quantum numbers, i.e. F, mF, F1, J, I1, I2, Ω, P.  
`UncoupledBasisState` is a class representing a TlF state with uncoupled quantum numbers, i.e. J, mJ, I1, m1, I2, m2, Ω, P.  
Finally `State` is a class representing a collection of states, since in most cases the TlF molecules are in a superposition state.

The states can be initialized manually, using 
```Python
import centrex_TlF as centrex
centrex.CoupledBasisState(F=1, mF=0, F1 = 1/2, I1 = 1/2, I2 = 1/2, Omega = 0, P = 1)
```
or using some of the functions to generate all hyperfine substates in a given J level: 
```Python
import centrex_TlF as centrex
centrex.states.generate_uncoupled_states_ground(Js = [0,1])
```

## `hamiltonian`
`hamiltonian` contains the functions to generate TlF hamiltonians in the X and B state in either coupled or uncoupled form.  
Generating a ground state X hamiltonian can be accomplished easily using some convenience functions:
```Python
import centrex_TlF as centrex

# generate the hyperfine sublevels in J=0 and J=1
QN = centrex.states.generate_uncoupled_states_ground(Js = [0,1])

# generate a dictionary with X hamiltonian terms
H = centrex.hamiltonian.generate_uncoupled_hamiltonian_X(QN)

# create a function outputting the hamiltonian as a function of E and B
Hfunc = centrex.hamiltonian.generate_uncoupled_hamiltonian_X_function(H)
```
All functions generating hamiltonians only require a list or array of TlF states. Generating the hamiltonian only for certain hyperfine sublevels is hence also straightforward.

To convert a hamiltonian from one basis to another transformation matrices can be calculated:
```Python
import centrex_TlF as centrex

# generate the hyperfine sublevels in J=0 and J=1
QN = centrex.states.generate_uncoupled_states_ground(Js = [0,1])
# generate the coupled hyperfine sublevels in J=0 and J=1
QNc = centrex.states.generate_coupled_states_ground(Js = [0,1])

# generate a dictionary with X hamiltonian terms
H = centrex.hamiltonian.generate_uncoupled_hamiltonian_X(QN)
Hfunc = centrex.hamiltonian.generate_uncoupled_hamiltonian_X_function(H)
H0 = Hfunc(E = [0,0,0], B = [0,0,1e-3])

# generate the transformation matrix
transform = centrex.hamiltonian.generate_transform_matrix(QN, QNc)

# calculate the transformed matrix
H0c = transform.conj().T@H0@transform
```
This is mostly used for optical bloch simulations where the coupled states representation is more convenient.

## `couplings`
`couplings` contains functions to generate coupling matrices and collapse matrices for optical bloch simulations, plus some other helpful tools.
