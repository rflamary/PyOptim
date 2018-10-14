""" Projection operator module """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np


def proj_pos(x, lambd=1, **kwargs):
    """
    positive orthant projection

    x : vector to project
    lambd: reg
    """
    return np.maximum(x, 0)


def proj_simplex(x, s, **kwargs):
    """
    projection on unit simplex

    x : vector to project
    s:
    """
    return s * projectUnitSimplex(x / s)


def projectUnitSimplex(v):
    Ne = len(v)
    meanv = float(sum(v)) / Ne
    tmp = np.array(v - np.tile(meanv - 1.0 / Ne, Ne))
    while np.any(tmp < 0):
        tmp[tmp < 0] = 0
        t = tmp[tmp > 0]
        t = t - (sum(t) - 1.0) / len(t)
        tmp[tmp > 0] = t
    return tmp


def projectUnitSimplexSparse(v, s, sparsityLevel):
    # sort element in reverse order. Keed the first 'sparsitylevel' components
    indices = np.argsort(v)[::-1][:sparsityLevel]
    tmp = s * projectUnitSimplex(v[indices] / s)
    proj = np.zeros(len(v))
    proj[indices] = tmp
    return proj


def proj_sparse_simplex(x, s, sparsitylevel, **kwargs):
    """
    sparse projection on unit simplex

    x : vector to project
    sparsitylevel: desired sparsity of the vector
    """
    return projectUnitSimplexSparse(x, s, sparsitylevel)
