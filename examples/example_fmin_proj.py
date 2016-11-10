# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:28:34 2016

@author: rflamary
"""


import numpy as np

import optim

n=2000
d=100

seed=1985
np.random.seed(seed)

# true model
wt=np.random.rand(d)
wt/=wt.sum()
w0=np.zeros((d,))



# data generation
sig=1.
X=np.random.randn(n,d)
y=np.dot(X,wt)+sig*np.random.randn(n)


# least square
wls=np.linalg.solve(X.T.dot(X),X.T.dot(y))
print("Err LS={}".format(optim.utils.norm(wt-wls)))

# optimization parameters
params=dict()
params['nbitermax']=1000
params['stopvarx']=1e-9
params['stopvarj']=1e-9
params['verbose']=True
params['bbrule']=False
params['log']=True

# Problem:
#   min_w |y-Xw|^2
#   s.t. |w|_1=1 and w>=0 (simplex)
#

# loss functions and regularization term
f=lambda w:optim.loss.loss_l2(w,X,y) # l2 loss
df=lambda w:optim.loss.grad_l2(w,X,y) # grad l2 loss
proj=lambda w: optim.proj.proj_simplex(w,1)

w,log=optim.solvers.fmin_proj(f,df,proj,w0,**params)
print("Err Simplex={}".format(optim.utils.norm(wt-w)))