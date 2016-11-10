
import numpy as np

def reg_l1(x,lambd=1,**kwargs):
    """
    l1 regularization (LASSO)

    x : vector to project
    lambd: regularization term
    """
    return lambd*np.sum(np.abs(x))


def prox_l1(x,lambd=1,**kwargs):
    """
    l1 regularization proximal operator (LASSO)

    x : vector to project
    lambd: regularization term
    """
    return np.sign(x)*np.maximum(np.abs(x)-lambd,0)

def reg_l2(x,lambd=1,**kwargs):
    """
    l2 regularization (ridge)

    x : vector to project
    lambd: regularization term
    """
    return lambd*np.sum(x**2)/2



def prox_l2(x,lambd=1,**kwargs):
    """
    l2 regularization proximity operator (ridge)

    x : vector to project
    lambd: reg
    """
    return x/(1+lambd)

