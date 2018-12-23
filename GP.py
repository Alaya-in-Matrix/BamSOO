import GPy
from GPyOpt.util.general import get_quantiles
import numpy as np
from math import pow, log, sqrt
import sys

# TODO: standardize the training data
class GP:
    def __init__(self, train_x, train_y, conf = {}):

        self.num_train = train_x.shape[0]
        if self.num_train < 2:
            print("At least two points are needed for GP modeling")
            sys.exit(1)

        self.ymean   = train_y.mean()
        self.ystd    = train_y.std()
        self.train_x = train_x.copy()
        self.train_y = (train_y - self.ymean) / self.ystd
        self.dim     = self.train_x.shape[1]
        self.rv      = conf.get('rv', 0.3);
        self.rl      = conf.get('rl', 0.3);

        kern   = GPy.kern.Matern52(input_dim = self.dim, ARD = True)
        self.m = GPy.models.GPRegression(self.train_x, self.train_y, kern)

        self.m.kern.variance       = self.rv * (self.train_y.max() - self.train_y.min())**2
        self.m.kern.lengthscale    = self.rl * (self.train_x.max(axis=0)  - self.train_x.min(axis=0))
        self.m.likelihood.variance = np.maximum(1e-6, 1e-2 * np.var(self.train_y))
        self.m.kern.lengthscale.set_prior(GPy.priors.LogGaussian(0, 1))

    def train(self):
        self.m.optimize(max_iters=200, messages=False)
        print(self.m.kern.lengthscale)
        
    def predict(self, x):
        if x.ndim < 2:
            x = x.reshape(1, self.dim)
        py, ps2 = self.m.predict(x)
        py  = self.ymean + py * self.ystd
        ps2 = ps2 * self.ystd**2
        return py, np.maximum(ps2, 0.0)

    def update_db(self, x, y):
        # self.ymean       = y.mean()
        # self.ystd        = y.std()
        self.train_x     = x.copy()
        self.train_y     = (y - self.ymean) / self.ystd
        self.m.set_XY(self.train_x, self.train_y)
