import numpy as np
import pandas as pd
import scipy.stats

# one-liners
mse = lambda y_true, y_pred : np.mean(np.square(y_true - y_pred))
mae = lambda y_true, y_pred : np.mean(np.abs(y_true - y_pred))
mape = lambda y_true, y_pred : np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100
volatility = lambda prices : np.square(prices[1:] - prices[:-1])
raw_return = lambda prices : (prices[1:] - prices[:-1]) / prices[:-1]
log_return = lambda prices : np.log(prices[1:] / prices[:-1])
raw_return_inv = lambda prev, ret : prev * (ret + 1)
log_return_inv = lambda prev, log_ret : prev * np.exp(log_ret)
autocorr = lambda x, q : np.array([np.corrcoef(x[k:], x[:-k])[0,1] for k in range(1,q+1)])


class Dataset:
    
    def __init__(self, ts, lags, shift=0.0):
        idx = np.unique(np.append(np.array(lags), 0))[::-1]        
        ds = np.array([ts[i-idx].tolist() for i in range(idx[0], ts.size)])
        self.X = ds[:,:-1]
        self.y = ds[:,-1] 
        self.max_val = ts.max()
        self.min_val = ts.min() 
        self.shift = shift
        self.ranges = np.array([self.X.min(axis=0), self.X.max(axis=0)]).T
    
    def get_scaled(self):
        X_scaled = (self.X - self.min_val) / (self.max_val - self.min_val) + self.shift
        y_scaled = (self.y - self.min_val) / (self.max_val - self.min_val) + self.shift
        ranges_scaled = (self.ranges - self.min_val) / (self.max_val - self.min_val) + self.shift
        return X_scaled, y_scaled , ranges_scaled              
        
    def get(self):
        return self.X, self.y, self.ranges
    
    def inv_scale(self, X):
        return self.min_val + (self.max_val - self.min_val)*X - self.shift
    
    
def load_raw_data(path):
    func = lambda x : time.mktime(dt.datetime.strptime(' '.join([x[0], x[1]]), "%d/%m/%y %H:%M:%S").timetuple())
    ts_data = pd.read_csv(path)
    ts_data.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'vol']
    ts_data.drop(['high', 'low', 'close', 'vol'], axis=1, inplace=True)
    ts_data['timestamp'] = ts_data.apply(func, axis=1)
    return ts_data


def autocorr_full(x, q = 0):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2+1:] / result[result.size//2]


def weiner(steps=1000, m=0, sigma=1):
    y = np.zeros(steps)
    for i in range(1,steps):
        y[i] = y[i-1] + m + sigma*np.random.randn()
    return y


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


def mark_anomaly(y_m, y_c, anomaly_thresh):
    return np.logical_and(y_m > 0, np.abs(y_c - y_m) > anomaly_thresh)

def feet_to_meters(y):
    return y * 0.3048

def meters_to_feet(y):
    return y / 0.3048