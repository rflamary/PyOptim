""" Generic optimization solvers """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License


import numpy as np

from .utils import armijo, norm, line_search_armijo


# format string for printing in verbose mode
prnt_str_name = "|{it:>5}|{loss:>13}|{dloss:>13}|{step:>13}|\n" \
                "|-----|-------------|-------------|-------------|"
prnt_str_loop = "|{it:5d}|{loss: 10e}|{dloss: 10e}|{step: 10e}|"


def fmin_prox(f, df, g, prox_g, x0, lambd=1., backtrack=True, nbitermax=1000,
              stopvarx=1e-9, stopvarj=1e-9, t0=10., verbose=False, m_back=1,
              sigma=1e-9, eta=2, nbitermax_back=100, bbrule=True, log=False,
              **kwargs):
    """ Minimize a sum of smooth and nonsmooth function using proximal splitting

    Solve the optimization problem:

        min_x f(x)+g(x)

        with:
        - f is differentiable (df) and Lipshictz gradient
        - g is non-differentiable but has a proximal operator (prox_g)

        The algorithm can use the bb_rule to update the step size

        Parameters:
            - backtrack : peform backtrack if true
            - bbrule : update step with bb rule
            - nbitermax : max number of iteratioin in algorithm
            - stopvarx : stopping criterion for relative variation of the
                norm of x
            - stopvarj : stopping criterion for relative variation of the cost
            - t0 : initial descent step
            - verbose : prinrt optimization information
            - m_back : window size for backtrack (if <1 then non decreasing)
            - sigma : descent parameter for backtrack
            - eta : value mitiplying t during backtrack
            - nbitermax_back : max number of backtrack iterations

    """
    x = x0.copy()

    grad = df(x, **kwargs)
    # grad[:]=0

    loss = list()
    loss.append(f(x, **kwargs) + g(x, lambd, **kwargs))

    if log:
        log = dict()
        log['loss'] = loss
    t = t0

    if verbose:
        print((prnt_str_name.format(it="It. ", loss='Loss ',
                                    dloss="Delta Loss ", step="Step ")))
        print((prnt_str_loop.format(it=0, loss=loss[-1], dloss=0, step=1 / t)))

    loop = True
    it = 1
    while loop:
        x_1 = x.copy()
        grad_1 = grad.copy()

        # gradient
        grad = df(x, **kwargs)

        # prox operator
        x = prox_g(x_1 - grad / t, lambd / t, **kwargs)

        # cost computation
        loss.append(f(x, **kwargs) + g(x, lambd, **kwargs))

        # line search backtrack
        it2 = 0
        thr_back = np.max([loss[-2 - k] - sigma / 2 * t * norm(x - x_1)
                           ** 2 for k in range(min(m_back, len(loss) - 1))])
        while loss[-1] > thr_back and it2 < nbitermax_back and backtrack:
            t = t * eta
            x = prox_g(x_1 - grad / t, lambd / t, **kwargs)
            loss[-1] = f(x, **kwargs) + g(x, lambd, **kwargs)
            thr_back = np.max([loss[-2 - k] - sigma / 2 * t * norm(x - x_1)
                               ** 2 for k in range(min(m_back, len(loss) - 1))
                               ])
            # print '\t',loss[-1],thr_back
            it2 += 1
        if it2 == nbitermax_back:
            print("Warning: backtrack failed")
        # print loss[-1],t

        # print information
        if verbose:
            if not (it) % 20:
                print((prnt_str_name.format(it="It. ", loss='Loss ',
                                            dloss="Delta Loss ",
                                            step="Step ")))
            print(prnt_str_loop.format(it=it,
                                       loss=loss[-1],
                                       dloss=(loss[-1] - loss[-2]
                                              ) / abs(loss[-2]),
                                       step=1 / t))

        # BB rule
        xbb = x - x_1
        ybb = grad - grad_1
        if it >= 1 and norm(xbb) > 1e-12 and norm(ybb) > 1e-12 and bbrule:
            t = abs(np.sum(xbb * ybb) / np.sum(xbb * xbb))
#        else:
#            t=t0

        # test convergence
        if norm(x - x_1) / norm(x) < stopvarx:
            loop = False
            if verbose:
                print("delta x convergence")
#
        if abs(loss[-1] - loss[-2]) / abs(loss[-2]) < stopvarj:
            loop = False
            if verbose:
                print("delta loss convergence")

        if it >= nbitermax:
            loop = False
            if verbose:
                print("Max number of iteration reached")

        # increment iteration
        it += 1

    if log:
        log['loss'] = loss
        return x, log
    else:
        return x


def fmin_proj(f, df, proj, x0, nbitermax=1000, stopvarx=1e-9, stopvarj=1e-9,
              t0=1., verbose=False, bbrule=True, log=False, **kwargs):
    """
    Solve the optimization problem:

        min_x f(x)

        s.t. s\in P

        with:
        - f is differentiable (df) and Lipshictz gradient
        - proj is a projection onto P

        The algorithm can use the bb_rule to update the step size

        Parameters:
            - backtrack : peform backtrack if true
            - bbrule : update step with bb rule
            - nbitermax : max number of iteratioin in algorithm
            - stopvarx : stopping criterion for relative variation of the
                norm of x
            - stopvarj : stopping criterion for relative variation of the cost
            - t0 : initial descent step
            - verbose : prinrt optimization information
            - m_back : window size for backtrack (if <1 then non decreasing)
            - sigma : descent parameter for backtrack
            - eta : value mitiplying t during backtrack
            - nbitermax_back : max number of backtrack iterations

    """
    x = x0.copy()

    grad = df(x, **kwargs)
    # grad[:]=0

    loss = list()
    deltax = list()

    loss.append(f(x, **kwargs))
    grad = df(x, **kwargs)
    deltax.append(
        np.linalg.norm(
            x -
            proj(
                x -
                grad,
                **kwargs),
            np.inf) /
        np.linalg.norm(
            x,
            np.inf))

    if log:
        log = dict()
        log['loss'] = loss
    t = t0

    if verbose:
        print(
            prnt_str_name.format(
                it="It. ",
                loss='Loss ',
                dloss="Delta Loss ",
                step="Step "))
        print(prnt_str_loop.format(it=0, loss=loss[-1], dloss=0, step=1 / t))

    def fproj(x):
        return f(proj(x, **kwargs), **kwargs)

    loop = True
    it = 1
    while loop:
        x_1 = x.copy()
        grad_1 = grad.copy()

        # gradient
        grad = df(x, **kwargs)

        # prox operator
        d = -grad / t
        x, tau, fnew = armijo(x_1, d, fproj, loss[-1], grad, tau=1, gamma=1e-6)

        x = proj(x)

        # cost computation
        loss.append(fnew)

        # line search backtrack

        # print loss[-1],t

        # print information
        if verbose:
            if not (it) % 20:
                print(
                    prnt_str_name.format(
                        it="It. ",
                        loss='Loss ',
                        dloss="Delta Loss ",
                        step="Step "))
            print(prnt_str_loop.format(it=it,
                                       loss=loss[-1],
                                       dloss=(loss[-1] - loss[-2]
                                              ) / abs(loss[-2]),
                                       step=tau / t))

        # BB rule
        xbb = x - x_1
        ybb = grad - grad_1
        if it >= 1 and norm(xbb) > 1e-12 and norm(ybb) > 1e-12 and bbrule:
            t = abs(np.sum(xbb * ybb) / np.sum(xbb * xbb))
        else:
            t = t0

        # test convergence
        deltax.append(
            np.linalg.norm(
                x -
                proj(
                    x -
                    grad,
                    **kwargs),
                np.inf) /
            np.linalg.norm(
                x,
                np.inf))
        if deltax[-1] < stopvarx:
            loop = False
            if verbose:
                print("delta x convergence")
#
        if abs(loss[-1] - loss[-2]) / abs(loss[-2]) < stopvarj:
            loop = False
            if verbose:
                print("delta loss convergence")

        if it >= nbitermax:
            loop = False
            if verbose:
                print("Max number of iteration reached")

        # increment iteration
        it += 1

    if log:
        log['loss'] = loss
        log['deltax'] = deltax
        return x, log
    else:
        return x


def fmin_cond(f, df, solve_c, x0, nbitermax=200,
              stopvarj=1e-9, verbose=False, log=False):
    """ F minimization with conditional gradient

        The function solves the following optimization problem:

    .. math::
        \min_x f(x)

        s.t. x\inc


    where :



    Parameters
    ----------

    x0 :  np.ndarray (ns,nt), optional
        initial guess (default is indep joint density)
    nbitermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    x : ndarray
        solution
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------



    """

    loop = 1

    if log:
        log = {'loss': []}

    x = x0
    f_val = f(x0)
    if log:
        log['loss'].append(f_val)

    it = 0

    if verbose:
        print(('{:5s}|{:12s}|{:8s}'.format(
            'It.', 'Loss', 'Delta loss') + '\n' + '-' * 32))
        print(('{:5d}|{:8e}|{:8e}'.format(it, f_val, 0)))

    while loop:

        it += 1
        old_fval = f_val

        # problem linearization
        g = df(x)

        # solve linearization
        xc = solve_c(x, g)

        deltax = xc - x

        # line search
        alpha, fc, f_val = line_search_armijo(f, x, deltax, g, f_val)

        x = x + alpha * deltax

        # test convergence
        if it >= nbitermax:
            loop = 0

        delta_fval = (f_val - old_fval) / abs(f_val)
        if abs(delta_fval) < stopvarj:
            loop = 0

        if log:
            log['loss'].append(f_val)

        if verbose:
            if it % 20 == 0:
                print(('{:5s}|{:12s}|{:8s}'.format(
                    'It.', 'Loss', 'Delta loss') + '\n' + '-' * 32))
            print(('{:5d}|{:8e}|{:8e}'.format(it, f_val, delta_fval)))

    if log:
        return x, log
    else:
        return x
