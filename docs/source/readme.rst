PyOptim Python numerical optimization toolbox
=============================================

This toolob aim at providing a unified interface to generic optimizers
for standard (LP,QP) and gradient based optimization problems (LBFGS,
Proximal Splitting, Projected gradient).

As of now it provides the following solvers:

-  Linear Program (LP) solver using scipy, cvxopt, or GUROBI solver.
-  Proximal spliting (a.k.a. ISTA) gradientd escnt for non smooth
   optimization.
-  Spectral Projected Gradient solvers (spectral is optionnal but
   strongly recommended).
-  Conditional gradient solver.

Planned integration are:

-  L-BFGS for smooth optimization (interface to scipy and others)
-  Stochastic gradients
-  Quadratic Program (QP) solvers

Installation
------------

The library has been tested on Linux, MacOSX and Windows.

The following Python modules are necessary and intalled automaticaly
with:

-  Numpy (>=1.11)
-  Scipy (>=1.0)
-  Cvxopt

The following modules are optional but obviously necessary to the
solvers using them:

-  Quadprog
-  Gurobipy (official Python interface of GUROBI)
-  Stdgrb

You can install the toolbox by downloading it and then running:

::

    python setup.py install --user # for user install (no root)

Contributors
------------

The main contributors of this module are:

-  `Rémi Flamary <http://remi.flamary.com/>`__
-  `Stéphane Canu <http://asi.insa-rouen.fr/enseignants/~scanu/>`__
