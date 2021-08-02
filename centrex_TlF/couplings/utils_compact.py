import numpy as np

def delete_row_column(arr, sl):
    sl = np.s_[sl.start:sl.stop-1]
    return np.delete(np.delete(arr, sl, axis = 0), sl, axis = 1)

def C_array_to_BR(C_array, Γ):
    return C_array.copy()**2 / Γ
    
def BR_to_C_array(BR, Γ):
    return np.sqrt(BR.copy()*Γ)

def compact_BR_array(BR_array, slice_compact):
    start, stop = slice_compact.start, slice_compact.stop
    new_shape = np.asarray(BR_array[0].shape) - (stop - start) + 1
    BR_array_new = []
    for BR in BR_array:
        id1,id2 = np.nonzero(BR)
        id1 = id1[0]
        id2 = id2[0]
        if (id1 not in range(start,stop)) & (id2 not in range(start,stop)):
            BR_array_new.append(delete_row_column(BR, slice_compact))
    
    BR_sum = np.zeros(new_shape, 'complex')
    BR_sum[start,:] = np.sum(BR_array,axis = 0)[slice_compact].sum(axis = 0)[-new_shape[0]:]
    for id1,id2 in zip(*np.nonzero(BR_sum)):
        BR_new = np.zeros(new_shape, 'complex')
        BR_new[id1,id2] = BR_sum[id1,id2]
        BR_array_new.append(BR_new)
    return np.array(BR_array_new)

def compact_C_array(C_array, Γ, slice_compact):
    BR = C_array_to_BR(C_array, Γ)
    BR_compact = compact_BR_array(BR, slice_compact)
    C_array_compact = BR_to_C_array(BR_compact, Γ)
    return C_array_compact