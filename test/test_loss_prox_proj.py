""" Test iterative solvers """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License


import numpy as np
import optim
# import pytest


def test_loss():

    n = 100
    d = 3

    np.random.seed(0)
    x = np.random.randn(n, d)
    y = 2 * np.random.randint(0, 2, size=(n,))
    x += y[:, None]

    w = np.zeros(d)

    optim.loss.loss_l2(w, x, y)
    optim.loss.grad_l2(w, x, y)

    optim.loss.loss_hinge2(w, x, y)
    optim.loss.grad_hinge2(w, x, y)

    optim.loss.loss_reglog(w, x, y)
    optim.loss.grad_reglog(w, x, y)


def test_proj():

    d = 10
    np.random.seed(0)
    w = np.random.randn(d)

    # test positivity
    w0 = optim.proj.proj_pos(w)
    np.testing.assert_array_less(-1e-300, w0)

    # pojection simplex
    w0 = optim.proj.proj_simplex(w, 1)
    np.testing.assert_almost_equal(1, w0.sum())
    w0 = optim.proj.proj_simplex(w, 10)
    np.testing.assert_almost_equal(10, w0.sum())

    # proj sparse simplex
    w0 = optim.proj.proj_sparse_simplex(w, 1, 3)
    np.testing.assert_almost_equal(1, w0.sum())
    np.testing.assert_almost_equal(3, (w0 > 0).sum())

    w0 = optim.proj.proj_sparse_simplex(w, 10, 3)
    np.testing.assert_almost_equal(10, w0.sum())
    np.testing.assert_almost_equal(3, (w0 > 0).sum())


if __name__ == "__main__":
    test_loss()
    test_proj()
    pass
