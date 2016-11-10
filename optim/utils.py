

import numpy as np

def norm(x):
    """l2 norm of vector (Frobenius for matrices)"""
    return np.sqrt(np.sum(x**2))


def min_interp_2(f0,fp0,x0,f1,x1):
    """
    minimize the second order polynomial with given values
    """

    A=np.array([[x0**2,x0,1],[2*x0,1,0],[x1**2,x1,1]])
    b=np.array([f0,fp0,f1])

    try:
        temp=np.linalg.solve(A,b)
        res=-temp[1]/2/temp[0]
    except np.linalg.LinAlgError:
        #print "Warning: second order lineserach "
        res=(x1+x0)/2


    return res



def armijo(x0,dx,f,f0,df0,tau=1,gamma=1e-4,nitermax=100,**kwargs):

    x = x0 + tau*dx;

    f_new=f(x,**kwargs);
    gtd=np.sum(df0*dx);


    it=0;

    while f_new>f0+gamma*tau*gtd or it==0:

        #temp = tau;

        tau=min_interp_2(f0,gtd,0,f_new,tau)
        tau=min(1,max(0,tau));

#        f_prev = f_new;
#        t_prev = temp;

        x = x0 + tau*dx;

        f_new = f(x,**kwargs);

        it=it+1;
        if it>nitermax:
            break
    #print it

    return x,tau,f_new