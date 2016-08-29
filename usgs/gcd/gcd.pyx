import cython
import numpy as np
cimport numpy as np

@cython.cdivision(True)
cdef int _gcd(int a, int b):
    cdef int r_0 = max(a, b)
    cdef int r_1 = min(a, b)
    cdef int r_2 = 1
    while r_2 > 0:
        r_2 = r_0 % r_1
        r_0 = r_1
        r_1 = r_2
    return r_0

@cython.wraparound(False)
@cython.boundscheck(False)
def gcd(np.ndarray[np.int64_t] a):
    cdef int d = a[0]
    cdef int i = 1
    for i in range(1, a.size):
        d = _gcd(d, a[i])
    return d