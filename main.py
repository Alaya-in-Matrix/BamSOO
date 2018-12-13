import SOO, BamSOO
import toml
import numpy as np

# # Rosenbrock
# def f(p):
#     x = p[0]
#     y = p[1]
#     return (1 - x)**2 + 100 * (y - x**2)**2

# conf              = dict()
# conf['lb']        = np.array([-5, -5])
# conf['ub']        = np.array([10, 10])
# conf['max_eval']  = 400
# conf['num_split'] = 2
# conf['rand_init'] = 20
# conf['eta']       = 0.1


# Shekel
def f(xx):
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
conf['max_eval']  = 1000
conf['num_split'] = 2
conf['rand_init'] = 100
conf['eta']       = 0.1

# conf           = toml.load('conf.toml');
opt            = BamSOO.BamSOO(f, conf);
best_x, best_y = opt.optimize();
print(best_y)

np.savetxt('dby', opt.dby)
np.savetxt('dbx', opt.dbx)
