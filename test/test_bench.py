""" Test iterative solvers """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License


import numpy as np
import optim
import time

d = 3


def get_data_0():

    c = np.array([2, 1, 0.5])
    A = np.array([[.5, .5, 0], [0, .5, .5]])
    b = np.array([1.0, 1.0])

    Aeq = np.ones((1, 3))
    beq = np.array([1.0])

    return c, A, b, Aeq, beq


def test_tic_toc():

    # use toc first : not cool!
    optim.bench.toc()

    #
    optim.bench.tic()
    time.sleep(.1)
    t = optim.bench.toq()

    np.testing.assert_almost_equal(t, .1, decimal=2)


def test_bench():

    def fun(x): return (time.sleep(x), 1.0)

    def gen(i, n): return ((n / 1000,), {})

    n = 10
    nbloop = 5

    res = optim.bench.bench_fun_val(fun, n, nbloop, gen, verbose='True')

    np.testing.assert_allclose(
        res['time'],
        np.arange(n) / 1000,
        rtol=1e-05,
        atol=1e-3)


if __name__ == "__main__":

    test_tic_toc()
    test_bench()
