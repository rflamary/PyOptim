""" Optimization benchmark loop """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

from __future__ import print_function

import numpy as np
import time

__time_tic_toc = time.time()


def tic():
    """ Python implementation of Matlab tic() function """
    global __time_tic_toc
    __time_tic_toc = time.time()


def toc(message='Elapsed time : {} s'):
    """ Python implementation of Matlab toc() function """
    t = time.time()
    print(message.format(t - __time_tic_toc))
    return t - __time_tic_toc


def toq():
    """ Python implementation of Julia toc() function """
    t = time.time()
    return t - __time_tic_toc


def bench_fun_val(fun, n, nbloop, gen, verbose='True'):
    """ optimization bench function """

    res = {}

    time_tot = np.zeros((nbloop, n))
    val_tot = np.zeros((nbloop, n))

    for i in range(nbloop):
        if verbose:
            print('Loop {}'.format(i))
        for j in range(n):
            params, kwparams = gen(i, j)

            tic()
            x, val_tot[i, j] = fun(*params, **kwparams)
            time_tot[i, j] = toq()

            if verbose:
                print('{:1.3f} '.format(time_tot[i, j]), end='')
        print()

    res = {'time_tot': time_tot,
           'val_tot': val_tot,
           'val': val_tot.mean(0),
           'time': time_tot.mean(0)}

    return res
