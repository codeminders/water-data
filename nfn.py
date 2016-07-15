import numpy as np
import time


class NFN:    
    
    def __init__(self, n_inputs, n_rules, x_range, step=1e-1, approximation='linear',
                 init='data', init_c=1.0, init_std=1.0, init_X=None, init_y=None):  
        self._m = n_rules + 2
        self._n = n_inputs
        
        if n_inputs != np.array(x_range).shape[0]:
            raise Exception('Input size is inconsistent with provided value ranges')
        
        grid = list()
        for rng in x_range:
            x_min = rng[0] - (rng[1] - rng[0]) / (n_rules + 1)
            x_max = rng[1] + (rng[1] - rng[0]) / (n_rules + 1)
            grid.append(np.linspace(x_min, x_max, n_rules + 4))
        grid = np.vstack(grid).T       
        a = grid[ :-2,:]
        b = grid[1:-1,:]
        c = grid[2:  ,:]
        self._step = step
        self._err = 0.0
        
        if approximation == 'linear':
            self._p = -1/(b - a)
            self._q = b
            self._r = np.ones(self._p.shape)
            self._mfunc = lambda z, p, q, r : np.maximum(0, p*np.abs(z - q) + r)            
        elif approximation == 'quadratic':
            self._p = -1 / (a - b) / (b - c)
            self._q = -self._p * (a + c)
            self._r = self._p * a * c
            self._mfunc = lambda z, p, q, r : np.maximum(0, p*np.square(z) + q*z + r)            
        elif approximation == 'gaussian':
            self._p = (b - a) / np.sqrt(-np.log(0.5))
            self._q = b
            self._r = np.ones(self._p.shape)
            self._mfunc = lambda z, s, m, r : np.exp(-np.square((z - m) / s))            
        else:
            raise Exception("Approximation method unknown")
         
        if init == 'data':
            if init_X is not None and init_y is not None:
                self._W = np.zeros((self._m, self._n))
                for i in range(self._n):
                    for j in range(self._m):
                        idx = np.logical_and(a[j,i] <= init_X[:,i], init_X[:,i] <= c[j,i])
                        if idx.sum() > 0:
                            self._W[j,i] = init_c * np.mean(init_y[idx]) / self._n
                        else:
                            self._W[j,i] = 0.0
            else:
                raise Exception("Initialization data must be provided for 'data' initialization method")
        elif init == 'random':
            self._W = init_c + init_std*np.random.randn(self._m, self._n)
        elif init == 'const':
            self._W = init_c * np.ones((self._m, self._n))
        else:
            raise Exception("Initialization method unknown")
        self._W_prev = np.array(self._W)
            
            
    def mfunc(self, x):
        N, n = x.shape
        if n != self._n:
            raise Exception("Input dimension mismatch")           
        p = np.broadcast_to(self._p, (N, self._m, self._n))
        q = np.broadcast_to(self._q, (N, self._m, self._n))
        r = np.broadcast_to(self._r, (N, self._m, self._n))        
        X = np.broadcast_to(x, (self._m, N, self._n))
        X = np.swapaxes(X, 0, 1)       
        self._M = self._mfunc(X, p, q, r)
        return self._M
    
        
    def predict(self, x):
        N, n = x.shape
        W = np.broadcast_to(self._W, (N, self._m, self._n))
        M = self.mfunc(x)
        Y = np.multiply(W, M)
        y = Y.sum(axis=(1,2))
        return y
    
    
    def fit(self, X_train, y_train, adaptive=True, inc=1.2, dec=0.5):
        N, n = X_train.shape
        if n != self._n:
            raise Exception("Input dimension mismatch")   
        if inc <= 1.0:
            raise Exception("Step increment should be > 1")
        if dec >= 1.0:
            raise Exception("Step decrement should be < 1")
            
        y = self.predict(X_train)  
        e = y_train - y
        err = np.square(e).mean()
        if adaptive:
            if err < self._err:
                self._W_prev = self._W
                self._step *= inc 
            else:
                self._W = self._W_prev
                self._step *= dec 
                
        dE_dM = np.broadcast_to(e, (self._n, self._m, N)) 
        dE_dM = np.swapaxes(dE_dM, 0, 2)
        dM_dW = self._M        
        dE_dW = dE_dM * dM_dW
        self._W += self._step * dE_dW.mean(axis = 0)   
        self._err = err
        return err
    
    
    def train(self, X_train, y_train, X_test, y_test, 
              n_epochs=10000, max_no_best=16, tol=1e-7, is_adaptive=True,
              verbose=100, return_errors=True, return_steps=True):
        if max_no_best < 1 and n_epochs < 1:
            raise Exception('Inconsistent stopping criteria')
   
        train_error = list()
        test_error = list()
        steps = list()
        
        best = (0, 1e+35, self._W)
        n_no_best = 0 
        i = 0
        
        if verbose > 0:
            print('Epoch\t| Train error\t| Test error\t| Step\t\t| Time\t\t|')
            print('--------+---------------+---------------+---------------+---------------+')
            
        def __print():
            if t < 10:
                print('%i\t| %.6f\t| %.6f\t| %.6f\t| %.3f\t\t|' % (i, train_err, test_err, self._step, t))
            else:
                print('%i\t| %.6f\t| %.6f\t| %.6f\t| %.3f\t|' % (i, train_err, test_err, self._step, t))       
        
        while i < n_epochs or n_epochs < 0:
            t = time.time()
            train_err = self.fit(X_train, y_train, adaptive=is_adaptive)
            y_test_pred = self.predict(X_test)
            test_err = np.square(y_test - y_test_pred).mean()
            train_error.append(train_err)
            test_error.append(test_err)
            steps.append(self._step)

            if max_no_best > 0:
                if test_err < best[1] - tol:
                    best = (i, test_err, self._W)
                    n_no_best = 0
                else:
                    n_no_best += 1                
                if n_no_best > max_no_best:
                    self._W = best[2]
                    t = time.time() - t
                    if i % verbose == 0 and verbose > 0:
                        __print()
                    break
                    
            t = time.time() - t
            if i % verbose == 0 and verbose > 0:
                __print()
            i += 1
            
        if return_errors and return_steps:
            return i, best[0], train_error, test_error, steps
        elif return_errors and not return_steps:
            return i, best[0], train_error, test_error
        elif not return_errors and return_steps:
            return i, best[0], steps
        else:
            return i, best[0]
    
    
class MultiplicativeNFN:    
    
    def __init__(self, n_inputs, n_rules, x_range, approximation='linear', init='random', init_mean=1.0, init_std=1.0):  
        self._m = n_rules + 2
        self._n = n_inputs
        
        grid = list()
        for rng in x_range:
            x_min = rng[0] - (rng[1] - rng[0]) / (n_rules + 1)
            x_max = rng[1] + (rng[1] - rng[0]) / (n_rules + 1)
            grid.append(np.linspace(x_min, x_max, n_rules + 4))
        grid = np.vstack(grid).T       
        a = grid[ :-2,:]
        b = grid[1:-1,:]
        c = grid[2:  ,:]
        
        if approximation == 'linear':
            self._p = -1/(b - a)
            self._q = b
            self._r = np.ones(self._p.shape)
            self._mfunc = lambda z, p, q, r : np.maximum(0, p*np.abs(z - q) + r)            
        elif approximation == 'quadratic':
            self._p = -1 / (a - b) / (b - c)
            self._q = -self._p * (a + c)
            self._r = self._p * a * c
            self._mfunc = lambda z, p, q, r : np.maximum(0, p*np.square(z) + q*z + r)            
        elif approximation == 'gaussian':
            self._p = (b - a) / np.sqrt(-np.log(0.5))
            self._q = b
            self._r = np.ones(self._p.shape)
            self._mfunc = lambda z, s, m, r : np.exp(-np.square((z - m) / s))            
        else:
            raise Exception("Approximation method unknown")
            
        if init == 'random':
            self._W = init_mean + init_std*np.random.randn(self._m, self._n)
        elif init == 'const':
            self._W = init_mean * np.ones((self._m, self._n))
        else:
            raise Exception("Initialization method unknown")
            
            
    def predict(self, x):
        N, n = x.shape
        if n != self._n:
            raise Exception("Input dimension mismatch")
            
        p = np.broadcast_to(self._p, (N, self._m, self._n))
        q = np.broadcast_to(self._q, (N, self._m, self._n))
        r = np.broadcast_to(self._r, (N, self._m, self._n)) 
        
        W = np.broadcast_to(self._W, (N, self._m, self._n))
        X = np.broadcast_to(x, (self._m, N, self._n))
        X = np.swapaxes(X, 0, 1)
        
        self._M = self._mfunc(X, p, q, r)
        Y = np.multiply(W, self._M)
        self._F = Y.sum(axis = 1)
        y = self._F.prod(axis = 1)
        return y
    
    
    def partial_fit(self, X_train, y_train, step = 1e+0):
        N, n = X_train.shape
        if n != self._n:
            raise Exception("Input dimension mismatch")

        y = self.predict(X_train)
        e = y_train - y

        E = np.broadcast_to(e, (self._n, self._m, N))        
        Y = np.broadcast_to(y, (self._n, self._m, N)) 
        F = np.broadcast_to(self._F, (self._m, N, self._n))
        
        E = np.swapaxes(E, 0, 2)      
        Y = np.swapaxes(Y, 0, 2)
        F = np.swapaxes(F, 0, 1)

        dE = E*Y*self._M/F
        self._W += step*dE.mean(axis = 0)        
        return np.square(e).mean()