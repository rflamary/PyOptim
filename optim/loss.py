""" Loss module """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License


import numpy as np

from .utils import norm

def loss_l2(w,X,y,**kwargs):
    """
    least square loss

    w : current model
    X : samples * variables matrix
    y : value to predict, samples vector

    """
    return norm(y-np.dot(X,w))**2/X.shape[0]/2

def grad_l2(w,X,y,**kwargs):
    """
    least square loss gradient

    w : current model
    X : samples * variables matrix
    y : value to predict, samples vector

    """
    return -1*np.dot(X.T,y-np.dot(X,w))/X.shape[0]

def loss_hinge2(w,X,y,**kwargs):
    """
    hinge loss

    w : current model
    X : samples * variables matrix
    y : value to predict, samples vector

    """
    err = np.maximum(0,1.-y*(np.dot(X,w)))
    return norm(err)**2/X.shape[0]/2

def grad_hinge2(w,X,y,**kwargs):
    """
    least square loss gradient

    w : current model
    X : samples * variables matrix
    y : value to predict, samples vector

    """
    err = np.maximum(0,1.-y*(np.dot(X,w)))
    return -1*np.dot(X.T,err*y)/X.shape[0]

def loss_reglog(w,X,y,**kwargs):
    """
    logistic regression loss

    w : current model
    X : samples * variables matrix
    y : value to predict, samples vector

    """
    p = np.exp(-y*(np.dot(X,w)))
    return np.sum(np.log(1.+p))/X.shape[0]

def grad_reglog(w,X,y,**kwargs):
    """
    logistic regression loss gradient

    w : current model
    X : samples * variables matrix
    y : value to predict, samples vector

    """
    p = np.exp(-y*(np.dot(X,w)))
    P=p/(1.+p)
    return -1*np.dot(X.T,P*y)/X.shape[0]