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

        self.mean        = np.mean(train_y);
        self.std         = np.std(train_y);
        self.train_x     = train_x.copy()
        self.train_y     = (train_y - self.mean) / self.std
        self.dim         = self.train_x.shape[1]
        self.min_trainig = self.dim + 1

        kern   = GPy.kern.Matern52(input_dim = self.dim, ARD = True)
        self.m = GPy.models.GPRegression(self.train_x, self.train_y, kern)

        
        self.m.kern.variance       = np.var(self.train_y)
        self.m.kern.lengthscale    = np.std(self.train_x, 0)
        self.m.likelihood.variance = np.maximum(2e-20, 1e-2 * np.var(self.train_y))
        self.m.likelihood.variance.constrain_bounded(1e-20, 1e10)

        # self.m.kern.variance.set_prior(GPy.priors.Gamma.from_EV(np.var(self.train_y), 120))
        # self.m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1e-2 * np.var(self.train_y), 4))
        # self.m.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(np.std(self.train_x, 0), 1000 * np.ones(np.std(self.train_x, 0).shape)))

        self.m.kern.variance.set_prior(GPy.priors.LogGaussian(0, 1))
        self.m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(0.02, 4))
        self.m.kern.lengthscale.set_prior(GPy.priors.LogGaussian(0, 10))

    def train(self):
        self.m.optimize(max_iters=200, messages=False)
        
    def predict(self, x):
        if x.ndim < 2:
            x = x.reshape(1, self.dim)
        py, ps2 = self.m.predict(x)
        py      = self.mean + py * self.std
        ps2     = ps2 * (self.std**2)
        return py, np.maximum(ps2, 0.0)

    def update_db(self, x, y):
        old_x        = self.train_x.copy()
        old_y        = self.train_y.copy()
        self.mean    = np.mean(y)
        self.std     = np.std(y)
        self.train_x = x.copy()
        self.train_y = (y - self.mean) / self.std
        self.m.set_XY(x, y)
