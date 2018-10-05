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
    

def lp_init_mat(c,A=None,b=None,Aeq=None,beq=None,lb=None,ub=None):
    
    n=c.shape[0]
    
    if A is None or b is None:
        A=np.zeros((0,n))
        b=np.zeros(0)
        
    if Aeq is None or beq is None:
        A=np.zeros((0,n))
        b=np.zeros(0)
        
    if lb is None:
        lb=-np.ones(n)*np.inf

    if ub is None:
        ub=np.ones(n)*np.inf
    
    return c, A, b, Aeq, beq, lb, ub


def lp_solve(c,A=None,b=None,Aeq=None,beq=None,lb=None,ub=None, solver='scipy',
             verbose=False, log=False, **kwargs):
    """ Solve the standard linear program
    
    Solve the following optimization problem:
        
    .. math::
        \min_x \quad x^Tc
        
        
        lb <= x <= ub
        
        Ax <= b
        
        Aeq x = beq
        
    All constraint parameters are optional, they will be ignore is left at 
    default value None
        
    
    Use the solver slected from : 
    
    
    Parameters
    ----------
    c : (d,) ndarray, float64
        Linear cost vector
    A : (n,d) ndarray, float64, optional
        Linear constraint matrix 
    b : (n,) ndarray, float64, optional
        Linear constraint vector
    Aeq : (n,d) ndarray, float64, optional
        Linear equality constraint matrix 
    beq : (n,) ndarray, float64, optional
        Linear equality constraint vector        
    lb : (d) ndarray, float64, optional
        Lower bound constraint        
    ub : (d) ndarray, float64, optional
        Upper bound constraint      
    solver : string, optional
        solver used to solve the linear program
        
    Returns
    -------
    x: (d,) ndarray
        Optimal solution x
    val: float
        optimal value of the objective (None if optimization error)
    log: dict
        Optional log output
    
    
    """
    
    
    if solver=='scipy':
        res=lp_solve_scipy(c,A,b,Aeq,beq,lb,ub,
             verbose=verbose,log=log, **kwargs)
    else:
        raise NotImplemented('Solver {} not implemented'.format(solver))
    
        
#    if not A.flags.c_contiguous:
#        A=A.copy(order='C')
        
    
    return res
        

def lp_solve_scipy(c,A,b,Aeq,beq,lb=None,ub=None,
             verbose=False, log=False, method='interior-point', **kwargs):
    
    n=c.shape[0]
    
    # handle the awfull bounds
    if lb is None:
        lb=-np.inf
        
    if ub is None:
        ub=np.inf    
        
    if isinstance(lb, (int, float, complex)) and isinstance(lb, (int, float, complex)):
        bounds=(lb,ub)
    else:
        
        
        if isinstance(lb, (int, float, complex)):
            lb=np.ones(n)*lb
            
        if isinstance(lb, (int, float, complex)):
            ub=np.ones(n)*ub

            
        bounds = [(ub[i],lb[i]) for i in range(n)]
            
    
    if 'disp' in kwargs:
        verbose= kwargs['disp'] or verbose
        del kwargs['disp']
            
    res=scipy.optimize.linprog(c,A,b,Aeq,beq,method=method,bounds=bounds,options={'disp':verbose,**kwargs})
    
    val= res.fun
    if not res.success:
        val= None

    if log:
        return res.x, val, res
    else:
        return res.x, val
    
    
    
    



def qp_solve(Q,c=None,A=None,b=None,lb=None,ub=None, method=-1,logtoconsole=1,crossover=-1):
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
    
    n=Q.shape[0]
    
    if c is None:
        c=np.zeros((n))
    
    if A is None or b is None:
        A=np.zeros((0,n))
        b=np.zeros(0)
        
    if lb is None:
        lb=-np.ones(n)*np.inf

    if ub is None:
        ub=np.ones(n)*np.inf
        
    if not A.flags.c_contiguous:
        A=A.copy(order='C')

    if not Q.flags.c_contiguous:
        Q=Q.copy(order='C')
        
    sol,val=qp_solve_0(Q,c,A,b,lb,ub,method,logtoconsole,crossover)
    

    return sol,val