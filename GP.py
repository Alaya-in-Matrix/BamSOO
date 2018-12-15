import GPy
from GPyOpt.util.general import get_quantiles
import numpy as np
from math import pow, log, sqrt
import sys

# TODO: standardize the training data
class GP:
    def __init__(self, train_x, train_y):

        self.num_train = train_x.shape[0]
        if self.num_train < 2:
            print("At least two points are needed for GP modeling")
            sys.exit(1)

        self.train_x     = train_x.copy()
        self.train_y     = train_y.copy()
        self.dim         = self.train_x.shape[1]
        self.min_trainig = self.dim + 1

        kern   = GPy.kern.Matern52(input_dim = self.dim, ARD = True)
        self.m = GPy.models.GPRegression(self.train_x, self.train_y, kern)

        # self.m.kern.variance       = 64
        # self.m.kern.lengthscale    = 3 * np.ones(self.dim)
        # self.m.likelihood.variance = np.maximum(2e-20, 1e-2 * np.var(self.train_y))
        # self.m.likelihood.variance.constrain_bounded(1e-20, 1e10)

        self.m.kern.variance       = (self.train_y.max() - self.train_y.min())**2
        self.m.kern.lengthscale    = 0.3 * (self.train_x.max(axis=0)  - self.train_x.min(axis=0))
        self.m.likelihood.variance = np.maximum(2e-20, 1e-2 * np.var(self.train_y))
        self.m.likelihood.variance.constrain_bounded(1e-20, 1e10)

        self.m.kern.variance.set_prior(GPy.priors.LogGaussian(np.log(self.m.kern.variance), 1))
        self.m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(0.02, 4))
        self.m.kern.lengthscale.set_prior(GPy.priors.LogGaussian(0, 10))

    def train(self):
        self.m.optimize(max_iters=100, messages=False)
        
    def predict(self, x):
        if x.ndim < 2:
            x = x.reshape(1, self.dim)
        py, ps2 = self.m.predict(x)
        return py, np.maximum(ps2, 0.0)

    def update_db(self, x, y):
        self.train_x = x.copy()
        self.train_y = y.copy()
        self.m.set_XY(self.train_x, self.train_y)
