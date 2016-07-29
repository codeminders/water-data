import numpy as np
from gcd import *

# one-liners
mse = lambda y_true, y_pred : np.mean(np.square(y_true - y_pred))
mae = lambda y_true, y_pred : np.mean(np.abs(y_true - y_pred))
mape = lambda y_true, y_pred : np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100


def autocorr_full(x, q = 0):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2+1:] / result[result.size//2]


def weiner(steps=1000, m=0, sigma=1):
    return np.cumsum(m + sigma*np.random.randn(steps))


def get_data(db, site_id):
    to_shift = {
        14137000 : -687,
        7377500  : +2,
        8186500  : +2,
        11456000 : +2
    }

    to_skip = {
        14178000: 1,
        50136400: 1,
        50075000: 1,
        50065500: 1,
        9481740:  1
    }
    
    data_m = db['measured'].find_one({'site_no': site_id})
    data_c = db['corrected'].find_one({'site_no': site_id})
    
    if data_m is None or data_c is None or site_id in to_skip:
        raise Exception("Site not found")
        
    Tm = np.array(data_m['utc'], dtype=np.int32)
    Zm = np.array(data_m['gh'],  dtype=np.float32)
    
    Tc = np.array(data_c['utc'], dtype=np.int32)
    Zc = np.array(data_c['gh'],  dtype=np.float32)
    
    if site_id in to_shift:
        Zm += to_shift[site_id]
        Zc += to_shift[site_id]
    
    return Tm, Zm, Tc, Zc 


def align_measurements(t_a, y_a, t_b, y_b, gcd_dt = True):
    dt_a = t_a[1:] - t_a[:-1]
    dt_b = t_b[1:] - t_b[:-1]
    
    dt_bu, n_b = np.unique(dt_b, return_counts = True)
    dt_au, n_a = np.unique(dt_a, return_counts = True)
    
    if gcd_dt:
        dt_u = np.hstack([dt_au, dt_bu]).astype(np.int64)
        dt = gcd(np.abs(dt_u[dt_u != 0]))
    else:
        dt = min(dt_bu[np.argmax(n_b)], dt_au[np.argmax(n_a)])
    
    offset_b = max(t_b[0] - t_a[0], 0)
    offset_a = max(t_a[0] - t_b[0], 0)
    N = (max(t_b[-1], t_a[-1]) - min(t_b[0], t_a[0])) // dt + 1
    
    y_b_new = np.zeros(N) - 1
    y_a_new = np.zeros(N) - 1
    
    idx_b = (t_b - t_b[0]) // dt
    idx_a = (t_a - t_a[0]) // dt
    
    y_b_new[idx_b] = y_b
    y_a_new[idx_a] = y_a
    
    return dt, y_a_new, y_b_new


def mark_anomaly(y_m, y_c, anomaly_thresh):
    return np.logical_and(y_m > 0, np.abs(y_c - y_m) > anomaly_thresh)

def feet_to_meters(y):
    return y * 0.3048

def meters_to_feet(y):
    return y / 0.3048