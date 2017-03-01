from math import inf

def find_sub(A, B):
    print(helper(A, B, 0, 0))

def helper(A, B, ai, bi):
    if ai == len(A): return 0
    NA =len(A)
    NB = len(B)
    m = inf
    max_gap = (NB-NA) - (bi-ai)
    for i in range(bi, bi+max_gap):
        m = min(abs(A[ai]-B[i])+helper(A, B, ai+1, bi+1+i), m)
    return m


def diff(A,B):
    import numpy  as np
    NA = len(A)
    NB = len(B)
    H = np.zeros((NA, NB))
    m = inf
    for i in reversed(range(NA)):
        for j in reversed(range(NB-1)):
            H[i,j] = abs(A[i] - B[j])
        m = min(H[i, :], m)
    print(H)
