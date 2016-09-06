import cython
import numpy as np
import numpy.ma as ma
cimport numpy as np


cdef extern from "complex.h":
    double complex I
    double complex csqrt(double complex z) nogil   
    double sqrt(double z) nogil   
    
cdef struct roots:
    double complex r0
    double complex r1
    double complex r2
    
ctypedef np.float64_t DTYPE_t


@cython.cdivision(True)
cdef roots solve_cubic(float a, float b, float c, float d):
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
        
    # Cardano's formula: r_i = -(b + C*k**i + D0 / (C*k**i)) / (3*a)
    res.r0 = -(b + C + D0 / C) / (3*a)            # i = 0
    res.r1 = -(b + C*k + D0 / (C*k)) / (3*a)      # i = 1
    res.r2 = -(b + C*k*k + D0 / (C*k*k)) / (3*a)  # i = 2
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
def bezier_spline(double x_a, double y_a, double dy_a, 
                  double x_b, double y_b, double dy_b, 
                  x, double eps = 1e-10):    
    cdef: 
        int n = x.size
        double x_1, x_2, y_1, y_2
        double c_3, c_2, c_1, c_0
        roots R
     
    if n == 2:
        By = np.array([y_a, 0.5*(y_a + y_b)])
    
    else:
        # cubic bezier curve
        x_1 = x[n // 3]  
        x_2 = x[n - n // 3] 
        y_1 = y_a + dy_a*(x_1 - x_a)        
        y_2 = y_b + dy_b*(x_2 - x_b)        
        t = np.zeros(n)
        
        c_3 = x_b - 3*x_2 + 3*x_1 - 1*x_a
        c_2 =       3*x_2 - 6*x_1 - 3*x_a 
        c_1 =               3*x_1 - 3*x_a 
        c_0 =                       1*x_a
        
        for i in range(n):
            c_0 = x_a - x[i]
            R = solve_cubic(c_3, c_2, c_1, c_0)
            if abs(R.r0.imag) < eps and -eps <= R.r0.real <= 1+eps:
                t[i] = R.r0.real                
            elif abs(R.r1.imag) < eps and -eps <= R.r1.real <= 1+eps:
                t[i] = R.r1.real                
            elif abs(R.r2.imag) < eps and -eps <= R.r2.real <= 1+eps:
                t[i] = R.r2.real                
            else:
                raise Exception("Bezier interpolation: no real solution found.")
            
        By = (1-t)**3 * y_a + 3*t*(1-t)**2 * y_1 + 3*(1-t)*t**2 * y_2 + t**3 * y_b
        
    return By


@cython.boundscheck(False)
@cython.wraparound(False)
def linear_interpolator(y, int max_gap=10000):
    cdef:
        int a = 0, b = 0, i = 1, gap_size = 0, n = y.size
    z = y.copy()
    for i in range(1, n):        
        if ~y.mask[i]:
            a = b; b = i  # a, b are non-negative gap bounds            
            if b-a > 1 and gap_size < max_gap:      
                fill = np.linspace(y.data[a], y.data[b], b-a+1)
                z.data[a:b+1] = fill
                z.mask[a:b+1] = False
            gap_size = 0            
        else:
            gap_size += 1
    return z


@cython.cdivision(True)
cdef double get_bisector(double x, double y, 
                         double xl, double yl, 
                         double xr, double yr):
    cdef double x1 = x - xl
    cdef double y1 = y - yl
    cdef double x1_n = sqrt(x1*x1 + y1*y1)

    cdef double x2 = xr - x
    cdef double y2 = yr - y
    cdef double x2_n = sqrt(x2*x2 + y2*y2)
    
    cdef double y3 = (y1 / x1_n + y2 / x2_n) / (x1 / x1_n + x2 / x2_n)
    return y3


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def bezier_interpolator(y, int max_gap=10000, int window_size=10):
    cdef:         
        double dy_a, dy_b
        int a = 0, b = 0, p = 0, q = 0, i = 0, j = 0, k = 0
        int n = y.size
        int gap_size = 0
    z = y.copy()
        
    for i in range(1, n):        
        if ~y.mask[i]:
            a = b; b = i  # a, b are non-negative gap bounds            
            if b-a > 1 and gap_size < max_gap:                
                # left bounding condition
                j = 0; p = 0
                while ( y.data[a-j] > 0 and a-j > 0 and j < window_size ) : j += 1
                if j <= 2:                    
                    while ( y.data[a-j-p] < 0 and j+p < max_gap and a-j-p > 0 ) : p += 1
                    dy_a = get_bisector(a, y.data[a], a-j-p, y.data[a-j-p], b, y.data[b])                     
                else:
                    dy_a = (y.data[a] - y.data[a-j+1]) / j  

                #right bounding condition
                k = 0; q = 0
                while ( y.data[b+k] > 0 and b+k < n-1 and k < window_size ) : k += 1                 
                if k <= 2:                    
                    while ( y.data[b+k+q] < 0 and k+q < max_gap and b+k+q < n-1 ) : q += 1
                    dy_b = get_bisector(b, y.data[b], a, y.data[a], b+k+q, y.data[b+k+q])                     
                else:
                    dy_b = (y.data[b+k-1] - y.data[b]) / k
            
                z[a:b] = bezier_spline(0, y.data[a], dy_a, b-a, y.data[b], dy_b, np.arange(b-a))

            gap_size = 0            
        else:
            gap_size += 1
                
    return z