from solve_cubic import *
import numpy as np


def bezier_interpolator(x_a, y_a, dy_a, x_b, y_b, dy_b, x, eps = 1e-10):
    if x.size == 2:
        By = np.array([y_a, 0.5*(y_a + y_b)])
    
    else:
        # cubic bezier curve
        x_1 = x[x.size // 3]  
        x_2 = x[x.size - x.size // 3] 
        y_1 = y_a + dy_a*(x_1 - x_a)        
        y_2 = y_b + dy_b*(x_2 - x_b)        
        t = np.zeros(x.size)
        
        coef = np.array([x_b - 3*x_2 + 3*x_1 - 1*x_a, 
                               3*x_2 - 6*x_1 - 3*x_a, 
                                       3*x_1 - 3*x_a, 
                                               1*x_a])
        for i in range(x.size):
            coef[3] = x_a - x[i]
            R = solve_cubic(coef[0], coef[1], coef[2], coef[3])

            if abs(R['r0'].imag) < eps and 0-eps <= R['r0'].real <= 1+eps:
                t[i] = R['r0'].real
                
            elif abs(R['r1'].imag) < eps and 0-eps <= R['r1'].real <= 1+eps:
                t[i] = R['r1'].real
                
            elif abs(R['r2'].imag) < eps and 0-eps <= R['r2'].real <= 1+eps:
                t[i] = R['r2'].real
                
            else:
                raise Exception("Bezier interpolation: no real solution found.")
            
        By = (1-t)**3 * y_a + 3*t*(1-t)**2 * y_1 + 3*(1-t)*t**2 * y_2 + t**3 * y_b
        
    return By


def get_bisector(x, y, xl, yl, xr, yr):
    x1 = x - xl
    y1 = y - yl
    x1_n = np.sqrt(x1**2 + y1**2)

    x2 = xr - x
    y2 = yr - y
    x2_n = np.sqrt(x2**2 + y2**2)
    
    y3 = (y1 / x1_n + y2 / x2_n) / (x1 / x1_n + x2 / x2_n)
    return y3


def fill_gaps(y, max_gap = 10000, spike_size=2, window_size=10):
    z = y.copy()
    a = b = 0
    gap_size = 0

    for i in range(1, y.size):
        
        if y[i] > 0:
            a = b; b = i  # a, b are non-negative gap bounds          
            
            if b - a > 1 and gap_size < max_gap:                
                # left bounding condition
                j = 0
                while ( y[a - j] > 0 and a - j > 0 and j < window_size ) : 
                    j += 1

                if j <= spike_size:
                    p = 0
                    while ( y[a - j - p] < 0 and j + p < max_gap ) : p += 1
                    dy_a = get_bisector(a, y[a], a-j-p, y[a-j-p], b, y[b])                     
                else:
                    p = 0
                    dy_a = (y[a] - y[a-j+1]) / j  

                #right bounding condition
                k = 0
                while ( y[b + k] > 0 and b + k < y.size-1 and k < window_size ) : 
                    k += 1     
                
                if k <= spike_size:    
                    q = 0
                    while ( y[b + k + q] < 0 and k + q < max_gap ) : q += 1
                    dy_b = get_bisector(b, y[b], a, y[a], b + k + q, y[b + k + q])                     
                else:
                    q = 0
                    dy_b = (y[b+k-1] - y[b]) / k
                    
                z[a:b] = bezier_interpolator(0, y[a], dy_a, b-a, y[b], dy_b, np.arange(b-a))
                
            gap_size = 0
            
        else:
            gap_size += 1
                
    return np.array(z)
