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

    Aeq = np.ones((1, 3))
    beq = np.array([1.0])

    return c, A, b, Aeq, beq


def valid_solver_const(f, **kwargs):
    """ valid and try all possible constrints on LP solver """

    thr = 1e-8

    c, A, b, Aeq, beq = get_data_0()

    lb = np.zeros(d)
    ub = np.ones(d)

    lsres = []

    # LP inneq const
    x, val0 = f(c, A, b, lb=lb, ub=ub, **kwargs)

    np.testing.assert_array_less(A.dot(x) - thr, b)
    lsres.append((x, val0))
    np.testing.assert_array_less(x - thr, ub)  # box constraint 1
    np.testing.assert_array_less(lb - thr, x)  # box constraint 12
    # LP inneq const  + box (given as values)

    x, val = f(c, A, b, lb=0.0, ub=1.0, **kwargs)

    np.testing.assert_array_less(A.dot(x) - thr, b)  # inequality constraint
    np.testing.assert_allclose(val, val0)  # equality constraints
    np.testing.assert_array_less(x - thr, ub)  # box constraint 1
    np.testing.assert_array_less(lb - thr, x)  # box constraint 12
    lsres.append((x, val))

    # LP inneq const + eq + box
    x, val = f(c, A, b, Aeq, beq, lb, ub, **kwargs)

    np.testing.assert_array_less(A.dot(x) - thr, b)  # inequality constraint
    np.testing.assert_array_less(val0 - thr, val)  # best loss is unconstrained
    np.testing.assert_allclose(Aeq.dot(x) - thr, beq)  # equality constraints
    np.testing.assert_array_less(x - thr, ub)  # box constraint 1
    np.testing.assert_array_less(lb - thr, x)  # box constraint 12
    lsres.append((x, val))
    #
    #

    # LP  eq + box
    x, val = f(c, Aeq=Aeq, beq=beq, lb=lb, ub=ub, **kwargs)

    np.testing.assert_allclose(Aeq.dot(x) - thr, beq)  # equality constraints
    np.testing.assert_array_less(x - thr, ub)  # box constraint 1
    np.testing.assert_array_less(lb - thr, x)  # box constraint 12
    lsres.append((x, val))

    return lsres


def test_lp_solve():

    l1 = valid_solver_const(optim.lp_solve, method='simplex')
    l2 = valid_solver_const(optim.lp_solve, method='interior-point')

    for ((temp, val1), (temp2, val2)) in zip(l1, l2):
        np.testing.assert_allclose(val1, val2, atol=1e-7)

@pytest.mark.skipif(not optim.stdsolvers.gurobipy, reason="gurobipy not installed")
def test_lp_gurobipy():

    l1 = valid_solver_const(optim.lp_solve)

    for i in range(-1, 6):
        l2 = valid_solver_const(optim.lp_solve, solver='gurobipy', method=i)

        for ((temp, val1), (temp2, val2)) in zip(l1, l2):
            np.testing.assert_allclose(val1, val2, atol=1e-7)


def test_lp_cvxopt():

    l1 = valid_solver_const(optim.lp_solve)

    for m in ['default', 'glpk']:
        l2 = valid_solver_const(optim.lp_solve, solver='cvxopt', method=m)

        for ((temp, val1), (temp2, val2)) in zip(l1, l2):
            np.testing.assert_allclose(val1, val2, atol=1e-7)

@pytest.mark.skipif(not optim.stdsolvers.stdgrb, reason="stdgrb not installed")
def test_lp_stdgrb():

    l1 = valid_solver_const(optim.lp_solve)

    for i in range(-1, 6):
        l2 = valid_solver_const(optim.lp_solve, solver='stdgrb', method=i)

        for ((temp, val1), (temp2, val2)) in zip(l1, l2):
            np.testing.assert_allclose(val1, val2, atol=1e-7)


if __name__ == "__main__":
    # execute only if run as a script
    import traceback
    import sys
    import code

#    try:

    test_lp_solve()
    test_lp_gurobipy()
    # test_lp_cvxopt()
    test_lp_stdgrb()
#    except:
#        type, value, tb = sys.exc_info()
#        traceback.print_exc()
#        last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
#        frame = last_frame().tb_frame
#        ns = dict(frame.f_globals)
#        ns.update(frame.f_locals)
#        code.interact(local=ns)
#
