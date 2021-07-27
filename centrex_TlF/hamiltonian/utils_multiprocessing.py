import numpy as np

def multi_transformation_matrix(i, state1, basis2):
    transform = np.zeros((len(basis2), len(basis2)))

    for j, state2 in enumerate(basis2):
        transform[i,j] = state1@state2

    return transform