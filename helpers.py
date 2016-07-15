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