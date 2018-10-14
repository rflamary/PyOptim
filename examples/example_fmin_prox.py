# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:28:34 2016

@author: rflamary
"""


import numpy as np
import optim

n = 100000
d = 1000

# true model
wt = np.zeros((d,))
wt[0] = 1
wt[1] = -1
w0 = np.zeros((d,))

seed = 1985
np.random.seed(seed)

# data generation
sig = 1.
X = np.random.randn(n, d)
y = np.dot(X, wt) + sig * np.random.randn(n)


# least square
wls = np.linalg.solve(X.T.dot(X), X.T.dot(y))
print("Err LS={}".format(optim.utils.norm(wt - wls)))

# optimization parameters
params = dict()
params['nbitermax'] = 1000
params['stopvarx'] = 1e-9
params['stopvarj'] = 1e-9
params['verbose'] = True
params['m_back'] = 1
params['bbrule'] = True
params['log'] = True

# loss functions and regularization term
# Problem:
#   min_w |y-Xw|^2+reg*|w|_1


def f(w): return optim.loss.loss_l2(w, X, y)  # l2 loss


def df(w): return optim.loss.grad_l2(w, X, y)  # grad l2 loss


g = optim.prox.reg_l1
prox_g = optim.prox.prox_l1

reg = 1e-2

w, log = optim.solvers.fmin_prox(f, df, g, prox_g, w0, lambd=reg, **params)
print("Err LASSO={}".format(optim.utils.norm(wt - w)))
