import numpy as np
from numpy.lib.stride_tricks import as_strided


def rolling_window_1d(a, window_size, stride = 1, padded = True):
    itemsize = a.dtype.itemsize
    k = window_size // 2
    if padded:
        output = np.hstack([np.zeros(k) + a[0], a, np.zeros(k + window_size % 2) + a[-1]])
    else:
        output = a
    return as_strided(output, 
                      strides = (stride*itemsize, itemsize), 
                      shape   = ((output.size - window_size) // stride + 1, window_size))


def roll_stat_1d(y, k, stat_func):
    samples = rolling_window_1d(y, k)
    return stat_func(samples, axis = 1)


def roll_dist_1d(y, kernel):
    n = kernel.size
    samples = rolling_window_1d(y, n) 
    a = samples.sum(axis = 1).reshape((-1,1))
    samples = samples / np.broadcast_to(a, samples.shape)
    K = np.broadcast_to(kernel, samples.shape)
    return np.sum(np.square(samples - K), axis = 1)


erosion_1d  = lambda y, k : roll_stat_1d(y, k, np.min)
dilation_1d = lambda y, k : roll_stat_1d(y, k, np.max)
roll_var_1d = lambda y, k : roll_stat_1d(y, k, np.var)
roll_std_1d = lambda y, k : roll_stat_1d(y, k, np.std)
median_filter_1d = lambda y, k : roll_stat_1d(y, k, np.median)


def box_filter_1d(y, k):
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode = 'same')


def gaussian_filter_1d(y, k, sigma):
    kernel = np.exp(-0.5*((np.arange(k) - k // 2) / s)**2) / s * 0.3989422804014327
    return np.convolve(y, kernel, mode = 'same')


def extreme_values(y, continuity, high = True, low = True):
    n = y.size // 2
    res = y.copy()
    
    z = np.sort(np.abs(y))
    dz = z[1:] - z[:-1]
    
    if low:
        mask_min = dz[:n][::-1] > continuity
        if mask_min.sum() > 0:
            i_min = n - np.argmax(mask_min)
            thr_min = 0.5*(z[i_min] + z[i_min - 1])
        else:
            thr_min = z[0] - 1
    else:
        thr_min = z[0] - 1
        
    if high:
        mask_max = dz[n:] > continuity
        if mask_max.sum() > 0:
            i_max = n + np.argmax(mask_max)
            thr_max = 0.5*(z[i_max] + z[i_max + 1])
        else:
            thr_max = z[-1] + 1   
    else:
        thr_max = z[-1] + 1  
    
    mask = (res < thr_min) | (res > thr_max)
    return mask


def detect_spikes(y, dt = 3600, min_amp = 0.5, thr = 0.25):
    dy_l = (y[1:-1] - y[:-2]) / dt * 3600
    dy_r = (y[1:-1] - y[2:]) / dt * 3600    
    abs_dy_l = np.abs(dy_l)
    abs_dy_r = np.abs(dy_r)    
    mask = ( dy_l*dy_r > 0 ) & \
           ( ( abs_dy_l > min_amp ) | ( abs_dy_r > min_amp ) ) & \
           ( np.abs(abs_dy_l - abs_dy_r) < thr*np.maximum(abs_dy_l, abs_dy_r) )
    return np.hstack([False, mask, False])


def detect_consts(y):
    dy = np.hstack([1, y[1:] - y[:-1]])
    return dy != 0
        

def fix_spikes(y, mask):
    z = y.copy()
    idx = np.where(mask)[0]
    z[idx] = 0.5*(y[idx - 1] + y[idx + 1])
    return z


def adaptive_filter_1d(y, max_kernel_size, filter_func):
    # Value-dependent kernel size: larger range of the sample 
    # maller the kernel. Assumes, that smaller ranges are actually
    # much smoother, than large ones.
    
    n = y.size
    k_max = max_kernel_size
    res = y.copy()
    b = y.min()
    a = np.log(k_max) / (y.max() - b)
    for i in range(k_max, n-k_max-1):
        sample = y[i-k_max:i+k_max+1]
        rng = sample.max() - sample.min()
        k = np.int32(k_max * np.exp(-a*(rng - b))) #
        res[i] = filter_func(y[i-k:i+k+1])
    return res


def find_zeros(y):
    mask = np.zeros(y.size)
    for i in range(1, y.size):
        y_i = y[i]
        y_j = y[i-1]
        if y_i > 0 and y_j > 0 or y_i <= 0 and y_j <= 0:
            continue
        elif y_i > 0 and y_j <= 0:
            mask[i] = +1
        else:
            mask[i] = -1
    return mask