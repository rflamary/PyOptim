""" Test iterative solvers """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License


import numpy as np
import optim


def test_fminprox():

    n = 1000
    d = 100

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

    # optimization parameters
    params = dict()
    params['nbitermax'] = 1000
    params['stopvarx'] = 1e-9
    params['stopvarj'] = 1e-9
    params['verbose'] = True
    params['m_back'] = 1
    params['bbrule'] = True
    params['log'] = True

    def f(w): return optim.loss.loss_l2(w, X, y)  # l2 loss

    def df(w): return optim.loss.grad_l2(w, X, y)  # grad l2 loss

    g = optim.prox.reg_l1
    prox_g = optim.prox.prox_l1

    reg = 1e-1

    w, val, log = optim.solvers.fmin_prox(
        f, df, g, prox_g, w0, lambd=reg, **params)
    print("Err LASSO={}".format(optim.utils.norm(wt - w)))

    assert np.sum(np.abs(w) > 0) == 2


def test_fminproj():

    n = 1000
    d = 100

    seed = 1985
    np.random.seed(seed)

    # true model
    wt = np.random.rand(d)
    wt /= wt.sum()
    w0 = np.zeros((d,))

    # data generation
    sig = 1.
    X = np.random.randn(n, d)
    y = np.dot(X, wt) + sig * np.random.randn(n)

    # optimization parameters
    params = dict()
    params['nbitermax'] = 1000
    params['stopvarx'] = 1e-9
    params['stopvarj'] = 1e-9
    params['verbose'] = True
    params['bbrule'] = False
    params['log'] = True

    # Problem:
    #   min_w |y-Xw|^2
    #   s.t. |w|_1=1 and w>=0 (simplex)
    #

    # loss functions and regularization term

    def f(w): return optim.loss.loss_l2(w, X, y)  # l2 loss

    def df(w): return optim.loss.grad_l2(w, X, y)  # grad l2 loss

    def proj(w): return optim.proj.proj_simplex(w, 1)

    w, val, log = optim.solvers.fmin_proj(f, df, proj, w0, **params)
    print("Err Simplex={}".format(optim.utils.norm(wt - w)))

    np.testing.assert_almost_equal(w.sum(), 1)


def test_fmincond():

    n = 1000
    d = 100

    seed = 1985
    np.random.seed(seed)

    # true model
    wt = np.random.rand(d)
    wt /= wt.sum()
    w0 = np.zeros((d,))

    # data generation
    sig = 1.
    X = np.random.randn(n, d)
    y = np.dot(X, wt) + sig * np.random.randn(n)

    params = dict()
    params['nbitermax'] = 1000
    params['stopvarj'] = 1e-9
    params['verbose'] = False
    params['log'] = True

    def f(w): return optim.loss.loss_l2(w, X, y)  # l2 loss

    def df(w): return optim.loss.grad_l2(w, X, y)  # grad l2 loss

    def solve_C(xk, g):  # solve the linearization
        v = np.zeros_like(xk)
        ag = np.abs(g)
        idx = np.argmax(ag)
        v[idx] = -np.sign(g[idx]) / idx.size
        return v

    w, val, log = optim.solvers.fmin_cond(f, df, solve_C, w0, **params)
    print("Err Simplex={}".format(optim.utils.norm(wt - w)))

    assert np.abs(w.sum()) <= 1
