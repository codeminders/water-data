def align_measurements(t_meas, y_meas, t_corr, y_corr):
    dt_corr, n_corr = np.unique(t_corr[1:] - t_corr[:-1], return_counts=True)
    dt_meas, n_meas = np.unique(t_meas[1:] - t_meas[:-1], return_counts=True)
    dt = min(dt_corr[np.argmax(n_corr)], dt_meas[np.argmax(n_meas)])
    
    offset_corr = max(t_corr[0] - t_meas[0], 0)
    offset_meas = max(t_meas[0] - t_corr[0], 0)
    N = (max(t_corr[-1], t_meas[-1]) - min(t_corr[0], t_meas[0])) // dt + 1
    
    y_corr_new = np.zeros(N) - 1
    y_meas_new = np.zeros(N) - 1
    
    idx_corr = np.cumsum((t_corr[1:] - t_corr[:-1]) // dt) + offset_corr // dt
    idx_meas = np.cumsum((t_meas[1:] - t_meas[:-1]) // dt) + offset_meas // dt
    
    y_corr_new[idx_corr] = y_corr[1:]
    y_meas_new[idx_meas] = y_meas[1:]
    
    y_corr_new[0] = y_corr[0]
    y_meas_new[0] = y_meas[0]
    
    return y_meas_new, y_corr_new


def bezier_interpolator(x_a, y_a, dy_a, x_b, y_b, dy_b, x):
    if x.size == 2:
        By = np.array([y_a, 0.5*(y_a + y_b)])
    
    else:
        # cubic bezier curve
        x_1 = x[x.size // 3]  
        x_2 = x[x.size - x.size // 3] 
        y_1 = y_a + dy_a*(x_1 - x_a)        
        y_2 = y_b + dy_b*(x_2 - x_b)        
        t = np.zeros(x.size)
        for i in range(x.size):
            r = np.roots([x_b - 3*x_2 + 3*x_1 - 1*x_a,
                                3*x_2 - 6*x_1 - 3*x_a,
                                        3*x_1 - 3*x_a,
                                                1*x_a - x[i]])
            idx = np.where(np.logical_and(r <= 1, r >= 0))[0][0]
            t[i] = r[idx]
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


def fix_sampling_rate(y, d, interpolation='const'):
    z = y.copy()
    k = 0
    flag = False
    for i in range(1, z.size):        
        if z[i] <= 0 and z[i-1] > 0:
            flag = True  
        if z[i] > 0 and z[i-1] > 0:
            flag = False
        if z[i] <= 0 and z[i-1] <= 0 and flag:
            k += 1            
        if z[i] > 0 and z[i-1] <= 0 and flag:
            if k <= d-1:
                if interpolation == 'const':
                    z[i-k-1:i] = z[i-k-2]
                elif interpolation == 'linear':
                    z[i-k-1:i] = z[i-k-2] + (z[i] - z[i-k-2])*np.linspace(0,1,k+3)[1:-1]
            else:
                flag = False
            k = 0
    return z


def fill_gaps(y, max_gap = 10000, window = 10):
    z = y.copy()
    a = b = c = d = 0
    k = 0
    flag = False
    
    for i in range(1, z.size): 
        
        if z[i] <= 0 and z[i-1] > 0:
            flag = True  
            
        elif z[i] > 0 and z[i-1] > 0:
            a = b; b = c; c = d; d = i
            flag = False
            
        elif z[i] <= 0 and z[i-1] <= 0 and flag:
            k += 1   
            
        elif z[i] > 0 and z[i-1] <= 0 and flag:
            a = b; b = c; c = d; d = i 
            
        if b != 0 and c-b > 1: 
            if k <= max_gap-1:
                if b-a > 1 or (b-a < 2 and z[a-1] < 0):
                    dy_b = get_bisector(b, z[b], a, z[a], c, z[c])
                else:
                    j = a-1
                    while a-j < window and z[j] > 0:
                        j -= 1
                    dy_b = (z[b] - z[max(a-j, 0)]) / j
                    
                if (d-c > 1) or (d-c < 2 and z[d+1] < 0):
                    dy_c = get_bisector(c, z[c], b, z[b], d, z[d])
                else:
                    j = d+1
                    while j-d < window and z[j] > 0:
                        j += 1
                    dy_c = (z[min(d+j, z.size-1)] - z[c]) / j
                    
                z[b:c] = bezier_interpolator(0, z[b], dy_b, c-b, z[c], dy_c, np.arange(c-b))
                
            else:
                flag = False
            k = 0
    return z