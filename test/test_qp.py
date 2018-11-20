""" Test iterative solvers """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License


import numpy as np
import optim
import pytest

d = 3


def get_data_0():

    c = np.array([2, 1, 0.5])
    A = np.array([[.5, .5, 0], [0, .5, .5]])
    b = np.array([1.0, 1.0])
    G = np.eye(3)
    G[-1] = 0.025

    Aeq = np.ones((1, 3))
    beq = np.array([1.0])

    return G, c, A, b, Aeq, beq


def valid_solver_const(f, **kwargs):
    """ valid and try all possible constrints on LP solver """

    thr = 1e-8

    G, c, A, b, Aeq, beq = get_data_0()

    lb = np.zeros(d)
    ub = np.ones(d)

    lsres = []

    # LP inneq const
    x, val0 = f(G, c, A, b, lb=lb, ub=ub, **kwargs)

    np.testing.assert_array_less(A.dot(x) - thr, b)
    lsres.append((x, val0))
    np.testing.assert_array_less(x - thr, ub)  # box constraint 1
    np.testing.assert_array_less(lb - thr, x)  # box constraint 12
    # LP inneq const  + box (given as values)

    x, val = f(G, c, A, b, lb=0.0, ub=1.0, **kwargs)

    np.testing.assert_array_less(A.dot(x) - thr, b)  # inequality constraint
    np.testing.assert_allclose(val, val0)  # equality constraints
    np.testing.assert_array_less(x - thr, ub)  # box constraint 1
    np.testing.assert_array_less(lb - thr, x)  # box constraint 12
    lsres.append((x, val))

    # LP inneq const + eq + box
    x, val = f(G, c, A, b, Aeq, beq, lb, ub, **kwargs)

    np.testing.assert_array_less(A.dot(x) - thr, b)  # inequality constraint
    np.testing.assert_array_less(val0 - thr, val)  # best loss is unconstrained
    np.testing.assert_allclose(Aeq.dot(x) - thr, beq)  # equality constraints
    np.testing.assert_array_less(x - thr, ub)  # box constraint 1
    np.testing.assert_array_less(lb - thr, x)  # box constraint 12
    lsres.append((x, val))
    #
    #

    # LP  eq + box
    x, val, log = f(G, c, Aeq=Aeq, beq=beq, lb=lb, ub=ub, log=True, **kwargs)

    np.testing.assert_allclose(Aeq.dot(x) - thr, beq)  # equality constraints
    np.testing.assert_array_less(x - thr, ub)  # box constraint 1
    np.testing.assert_array_less(lb - thr, x)  # box constraint 12
    lsres.append((x, val))

    return lsres


def test_qp_solve():

    valid_solver_const(optim.qp_solve)


@pytest.mark.skipif(not optim.stdsolvers.quadprog,
                    reason="quadprog not installed")
def test_qp_quadprog():

    l1 = valid_solver_const(optim.qp_solve)
    l2 = valid_solver_const(optim.qp_solve, solver='quadprog')

    for ((temp, val1), (temp2, val2)) in zip(l1, l2):
        np.testing.assert_allclose(val1, val2, atol=1e-7)


@pytest.mark.skipif(not optim.stdsolvers.gurobipy,
                    reason="gurobipy not installed")
def test_qp_gurobipy():

    l1 = valid_solver_const(optim.qp_solve)
    l2 = valid_solver_const(optim.qp_solve, solver='gurobipy')

    for ((temp, val1), (temp2, val2)) in zip(l1, l2):
        np.testing.assert_allclose(val1, val2, atol=1e-7)

    for i in range(-1, 3):
        l2 = valid_solver_const(optim.qp_solve, solver='gurobipy', method=i)

        for ((temp, val1), (temp2, val2)) in zip(l1, l2):
            np.testing.assert_allclose(val1, val2, atol=1e-7)


if __name__ == "__main__":

    test_qp_solve()
    test_qp_quadprog()
    test_qp_gurobipy()
#
