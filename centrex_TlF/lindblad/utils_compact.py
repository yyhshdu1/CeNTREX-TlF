from centrex_TlF.couplings.collapse import collapse_matrices
import numpy as np
__all__ = [
    ''
]

def delete_row_column_symbolic(arr, sl):
    arr_copy = arr.copy()
    sl = np.s_[sl.start:sl.stop-1]
    deleted = 0
    for idx in range(sl.start, sl.stop):
        arr_copy.row_del(idx-deleted)
        arr_copy.col_del(idx-deleted)
        deleted += 1
    return arr_copy