import SOO as opt
import toml
import numpy as np

def f(x):
    return x**2;

conf           = toml.load('conf.toml');
opt            = SOO(f, conf);
best_x, best_y = opt.optimize();
