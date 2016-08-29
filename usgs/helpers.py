import numpy as np
import numpy.ma as ma
from usgs.gcd.gcd import *

# one-liners
mse  = lambda y_true, y_pred : np.mean(np.square(y_true - y_pred))
mae  = lambda y_true, y_pred : np.mean(np.abs(y_true - y_pred))
mxae = lambda y_true, y_pred : np.max(np.abs(y_true - y_pred)) # MaXimum Absolute Error
mape = lambda y_true, y_pred : np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100
rmse = lambda y_true, y_pred : np.sqrt(np.mean(np.square(y_true - y_pred)))

ft_to_m = lambda y : y * 0.3048
m_to_ft = lambda y : y / 0.3048


to_shift = {
    14137000 : -687,
    7377500  : +2,
    8186500  : +2,
    11456000 : +2
}

to_skip = [14178000, 50136400, 50075000, 50065500, 9481740]


class TS:    
    def __init__(self, time, values):
        self.t = time
        self.x = values


def get_data(db, site_id):    
    data_m = db['measured'].find_one({'site_no': site_id})
    data_c = db['corrected'].find_one({'site_no': site_id})
    
    if data_m is None or data_c is None or site_id in to_skip:
        raise Exception("Site not found")
        
    y_m = TS(np.array(data_m['utc'], dtype = np.int32),
             np.array(data_m['gh'],  dtype = np.float32)
    
    y_c = TS(np.array(data_c['utc'], dtype = np.int32), 
             np.array(data_c['gh'],  dtype = np.float32)
    
    if site_id in to_shift:
        y_m['y'] += to_shift[site_id]
        y_c['y'] += to_shift[site_id]
    
    return y_m, y_c


def align_measurements(X, Y, gcd_dt = True):
    dt_X = X.t[1:] - X.t[:-1]
    dt_Y = Y.t[1:] - Y.t[:-1]
    
    dt_X_u, n_X = np.unique(dt_X, return_counts = True)
    dt_Y_u, n_Y = np.unique(dt_Y, return_counts = True)
    
    if gcd_dt:
        dt_u = np.hstack([dt_X_u, dt_Y_u]).astype(np.int64)
        dt = gcd(np.abs(dt_u[dt_u != 0]))
    else:
        dt = min(dt_X_u[np.argmax(n_X)], dt_Y_u[np.argmax(n_Y)])
    
    offset_X = max(Y.t[0] - X.t[0], 0)
    offset_Y = max(X.t[0] - Y.t[0], 0)
    N = (max(X.t[-1], Y.t[-1]) - min(X.t[0], Y.t[0])) // dt + 1
    
    X_new = np.zeros(N) - 1
    Y_new = np.zeros(N) - 1
    
    X_idx = (X.t - X.t[0] + offset_X) // dt
    Y_idx = (Y.t - Y.t[0] + offset_Y) // dt
    
    X_new[X_idx] = X.x
    Y_new[Y_idx] = Y.x
    
    return dt, ma.masked_less(X_new, 0), ma.masked_less(Y_new, 0)