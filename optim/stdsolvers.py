""" Generic optimization solvers """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
import cvxopt
import scipy.optimize

try:
    import stdgrb
except ImportError:
    stdgrb = False

try:
    import gurobipy as gurobipy
except ImportError:
    gurobipy = False


def lp_init_mat(c, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None):
    """ initialize all matrices for lp problem with correct size"""

    n = c.shape[0]

    if A is None or b is None:
        A = np.zeros((0, n))
        b = np.zeros(0)

    if Aeq is None or beq is None:
        Aeq = np.zeros((0, n))
        beq = np.zeros(0)

    if lb is None:
        lb = -np.ones(n) * np.inf

    if ub is None:
        ub = np.ones(n) * np.inf

    if isinstance(lb, (int, float, complex)):
        lb = np.ones(n) * lb

    if isinstance(ub, (int, float, complex)):
        ub = np.ones(n) * ub

    return c, A, b, Aeq, beq, lb, ub


def lp_solve(c, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None, solver='scipy',
             verbose=False, log=False, **kwargs):
    """ Solve a standard linear program with linear constraints

    Solve the following optimization problem:

    .. math::
        \min_x \quad x^Tc


        lb <= x <= ub

        Ax <= b

        A_{eq} x = b_{eq}

    Return val as None if optmization failed.

    All constraint parameters are optional, they will be ignore is left at
    default value None.


    Use the solver selected from (default 'scipy'):
        - 'scipy'
            scipy.optimize solver with interior point solver and
            available methods 'interior-point' or 'simplex'
        - 'cvxopt'
            cvxopt interior point solver ('default') with other
            available methods 'mosek' or 'glpk'
        - 'gurobipy'
            gurobi solver with official python interface
        - 'stdgurobi'
            gurobi solver with c interface more efficient for dense problems


    Parameters
    ----------
    c : (d,) ndarray, float64
        Linear cost vector.
    A : (n,d) ndarray, float64, optional
        Linear inequality constraint matrix.
    b : (n,) ndarray, float64, optional
        Linear inequality constraint vector.
    Aeq : (n,d) ndarray, float64, optional
        Linear equality constraint matrix .
    beq : (n,) ndarray, float64, optional
        Linear equality constraint vector.   .
    lb : (d) ndarray, float64, optional
        Lower bound constraint, -np.inf if not provided.
    ub : (d) ndarray, float64, optional
        Upper bound constraint, np.inf if not provided.
    solver : string, optional
        Select solver used to solve the linear program from 'scipy', 'cvxopt'
        'gurobipy', 'stdgrb'.
    verbose : boolean, optional
        Print optimization informations.
    log : boolean, optional
        Return a dictionary with optim informations in adition to x and val

    Returns
    -------
    x: (d,) ndarray
        Optimal solution x
    val: float
        optimal value of the objective (None if optimization error)
    log: dict
        Optional log output


    """

    if solver == 'scipy':
        res = lp_solve_scipy(c, A, b, Aeq, beq, lb, ub,
                             verbose=verbose, log=log, **kwargs)
    elif solver == 'stdgrb':
        res = lp_solve_stdgrb(c, A, b, Aeq, beq, lb, ub,
                              verbose=verbose, log=log, **kwargs)
    elif solver == 'gurobipy':
        res = lp_solve_gurobipy(c, A, b, Aeq, beq, lb, ub,
                                verbose=verbose, log=log, **kwargs)
    elif solver == 'cvxopt':
        res = lp_solve_cvxopt(c, A, b, Aeq, beq, lb, ub,
                              verbose=verbose, log=log, **kwargs)
    else:
        raise NotImplementedError('Solver {} not implemented'.format(solver))

    return res


def lp_solve_scipy(c, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None,
                   verbose=False, log=False, method='interior-point', **kwargs):

    n = c.shape[0]

    # handle the awfull bounds
    if lb is None:
        lb = -np.inf

    if ub is None:
        ub = np.inf

    if isinstance(lb, (int, float, complex)) and isinstance(
            lb, (int, float, complex)):
        bounds = (lb, ub)
    else:

        if isinstance(lb, (int, float, complex)):
            lb = np.ones(n) * lb

        if isinstance(lb, (int, float, complex)):
            ub = np.ones(n) * ub

        bounds = [(ub[i], lb[i]) for i in range(n)]

    # catch is disp was given to true (scipy interfface compativility)
    if 'disp' in kwargs:
        verbose = kwargs['disp'] or verbose
        del kwargs['disp']

    res = scipy.optimize.linprog(c, A, b, Aeq, beq, method=method, bounds=bounds, options={'disp': verbose, **kwargs})

    # check if sucessful
    val = res.fun
    if not res.success:
        val = None

    if log:
        return res.x, val, res
    else:
        return res.x, val


def lp_solve_stdgrb(c, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None,
                    verbose=False, log=False, method='default', crossover=-1, **kwargs):

    if not stdgrb:
        raise ImportError("stdgrb not installed")

    c, A, b, Aeq, beq, lb, ub = lp_init_mat(c, A, b, Aeq, beq, lb, ub)

    method_to_int = {'default': -1,
                     'primal-simplex': 0,
                     'dual-simplex': 1,
                     'barrier': 2,
                     'concurrent': 3,
                     'deterministic-concurrent': 4,
                     'deterministic-concurrent-simplex': 5}

    if method in method_to_int:
        method = method_to_int[method]

    # add equality as two inequality (stdgrb do not handle that well yet)
    A2 = np.concatenate((A, Aeq, -Aeq), 0)
    b2 = np.concatenate((b, beq, -beq))

    sol, val = stdgrb.lp_solve(c, A2, b2, lb=lb, ub=ub, method=method,
                               logtoconsole=verbose, crossover=crossover)

    res = {'x': sol, 'fun': val, 'success': val is not None}

    if log:
        return sol, val, res
    else:
        return sol, val


def lp_solve_gurobipy(c, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None,
                      verbose=False, log=False, method='default', crossover=-1, **kwargs):

    if not gurobipy:
        raise ImportError("gurobipy not installed")

    n = c.shape[0]

    method_to_int = {'default': -1,
                     'primal-simplex': 0,
                     'dual-simplex': 1,
                     'barrier': 2,
                     'concurrent': 3,
                     'deterministic-concurrent': 4,
                     'deterministic-concurrent-simplex': 5}

    if method in method_to_int:
        method = method_to_int[method]

    m = gurobipy.Model("LP")

    # set paparemters
    m.Params.Method = method
    m.Params.LogToConsole = verbose
    m.Params.Crossover = crossover

    x = m.addVars(n, lb=lb, ub=ub, name="x")

    m.setObjective(gurobipy.quicksum((c[i] * x[i]
                                      for i in range(n))), gurobipy.GRB.MINIMIZE)

    if A is not None and b is not None:
        m.addConstrs((gurobipy.quicksum((x[j] * A[i, j]
                                         for j in range(n) if A[i, j])) <= b[i]
                      for i in range(A.shape[0])), "Ax<=b")

    if Aeq is not None and beq is not None:
        m.addConstrs((gurobipy.quicksum((x[j] * Aeq[i, j]
                                         for j in range(n) if Aeq[i, j])) == beq[i]
                      for i in range(Aeq.shape[0])), "Aeq x=beq")
    # add equality as two inequality (stdgrb do not handle that well yet)

    # m.update()
    try:

        m.optimize()

        sol = np.zeros_like(c)
        for i in range(n):
            sol[i] = m.getVars()[i].x
        val = m.getObjective().getValue()
        res = {'x': sol, 'fun': val, 'success': val is not None}

    except gurobipy.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    if log:
        return sol, val, res
    else:
        return sol, val


def lp_solve_cvxopt(c, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None,
                    verbose=False, log=False, method='default', **kwargs):

    n = c.shape[0]

    c, A, b, Aeq2, beq2, lb2, ub2 = lp_init_mat(c, A, b, Aeq, beq, lb, ub)

    mat = cvxopt.matrix

    A2 = A
    b2 = b

    # add constraints into A matrix
    if ub is not None:
        A2 = np.concatenate((A2, np.eye(n)), 0)
        b2 = np.concatenate((b2, ub2))
    if lb is not None:
        A2 = np.concatenate((A2, -np.eye(n)), 0)
        b2 = np.concatenate((b2, -lb2))

    c = mat(c)
    A2 = mat(A2)
    b2 = mat(b2)

    Aeq = mat(Aeq) if Aeq is not None else None
    beq = mat(beq) if beq is not None else None

    if method == 'default':
        method = None

    # add equality as two inequality (stdgrb do not handle that well yet)
    if Aeq is not None and beq is not None:
        A2 = np.concatenate((A, Aeq, -Aeq), 0)
        b2 = np.concatenate((b, beq, -beq))

    cvxopt.solvers.options['show_progress'] = verbose

    res = cvxopt.solvers.lp(c, A2, b2, Aeq, beq, solver=method, **kwargs)

    #res={'x':sol,'fun':val,'success': val is not None}
    # print(res)

    for key in ['x', 'y', 's', 'z']:
        res[key] = np.array(res[key]).ravel()

    if log:
        return res['x'], res['primal objective'], res
    else:
        return res['x'], res['primal objective']


def qp_solve(Q, c=None, A=None, b=None, lb=None, ub=None,
             method=-1, logtoconsole=1, crossover=-1):
    """ Solves a standard quadratic program

    Solve the following optimization problem:

    .. math::
        \min_x  x^TQx+x^Tc

        s.t.

        lb <= x <= ub

        Ax <= b


    Uses the gurobi solver.

    Parameters
    ----------
    Q : (d,d) ndarray, float64, optional
        Quadratic cost matrix matrix
    c : (d,) ndarray, float64
        Linear cost vector
    A : (n,d) ndarray, float64, optional
        Linear constraint matrix
    b : (n,) ndarray, float64, optional
        Linear constraint vector
    lb : (d) ndarray, float64, optional
        Lower bound constraint
    ub : (d) ndarray, float64, optional
        Upper bound constraint
    method : int, optional
        Selected solver from
        * -1=automatic (default),
        * 0=primal simplex,
        * 1=dual simplex,
        * 2=barrier,
        * 3=concurrent,
        * 4=deterministic concurrent,
        * 5=deterministic concurrent simplex
    logtoconsole : int, optional
        If 1 the print log in console,
    crossover : int, optional
        Select crossover strategy for interior point (see gurobi documentation)

    Returns
    -------
    x: (d,) ndarray
        Optimal solution x
    val: float
        optimal value of the objective (None if optimization error)


    """

    n = Q.shape[0]

    if c is None:
        c = np.zeros((n))

    if A is None or b is None:
        A = np.zeros((0, n))
        b = np.zeros(0)

    if lb is None:
        lb = -np.ones(n) * np.inf

    if ub is None:
        ub = np.ones(n) * np.inf

    if not A.flags.c_contiguous:
        A = A.copy(order='C')

    if not Q.flags.c_contiguous:
        Q = Q.copy(order='C')

    sol, val = qp_solve_0(Q, c, A, b, lb, ub, method, logtoconsole, crossover)

    return sol, val
