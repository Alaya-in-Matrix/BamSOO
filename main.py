import SOO
import toml
import numpy as np

def f(p):
    x = p[0]
    y = p[1]
    return (1 - x)**2 + 100 * (y - x**2)**2


def shekel(xx):
    m = 10;
    b = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5]);
    C = [[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0], 
         [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6], 
         [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0], 
         [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]]

    outer = 0;
    for ii in range(m):
            bi = b[ii];
            inner = 0;
            for jj in range(4):
                xj = xx[jj];
                Cji = C[jj][ii];
                inner = inner + (xj-Cji)**2;
            outer = outer + 1/(inner+bi);
    y = -outer;
    return y

conf              = dict()
conf['lb']        = np.zeros(4)
conf['ub']        = 10 * np.ones(4)
conf['max_eval']  = 700
conf['num_split'] = 2

# conf           = toml.load('conf.toml');
opt            = SOO.SOO(shekel, conf);
best_x, best_y = opt.optimize();
print(best_y)

np.savetxt('dby', opt.dby)
np.savetxt('dbx', opt.dbx)
