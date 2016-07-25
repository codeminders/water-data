import cython

cdef extern from "complex.h":
    double complex I
    double complex csqrt(double complex z) nogil   

cdef struct roots:
    double complex r0
    double complex r1
    double complex r2

@cython.cdivision(True)
def solve_cubic(float a, float b, float c, float d):
    cdef roots res
    if a == 0 and b == 0:
        res.r0 = <double complex>(-d/c)
        res.r1 = <double complex>(-d/c)
        res.r1 = <double complex>(-d/c)
        return res
    
    cdef:
        double D0 = b*b - 3*a*c
        double D1 = 2*b*b*b - 9*a*b*c + 27*a*a*d
        double D2 = D1*D1 - 4*D0*D0*D0
        double complex D3 = csqrt(D2)   
        double complex D4 = D1 + D3 
        double complex D5 = D4 if D4 != 0 else D1 - D3    
        double complex C = (0.5*D5) ** 0.3333333333333333
        double complex k = -0.5 - 0.8660254037844386j # -1/2 - 1/2*np.sqrt(3)j
        
    # r_i = -(b + C*k**i + D0 / (C*k**i)) / (3*a)
    res.r0 = -(b + C + D0 / C) / (3*a)            # i = 0
    res.r1 = -(b + C*k + D0 / (C*k)) / (3*a)      # i = 1
    res.r2 = -(b + C*k*k + D0 / (C*k*k)) / (3*a)  # i = 2
    return res
