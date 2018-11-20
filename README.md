# PyOptim Python numerical optimization toolbox

[![Build Status](https://travis-ci.org/rflamary/PyOptim.svg?branch=master)](https://travis-ci.org/rflamary/PyOptim)
[![Documentation Status](https://readthedocs.org/projects/pyoptim/badge/?version=latest)](https://pyoptim.readthedocs.io/en/latest/?badge=latest)

This toolob aim at providing a unified interface to generic optimizers 
for standard (LP,QP) and gradient based optimization problems  (LBFGS, 
Proximal Splitting, Projected gradient). 

As of now it provides the following solvers:

* Linear Program (LP) solver using scipy, cvxopt, or GUROBI solver.
* Quadratic Program (QP) solvers using cvxopt, guroby or quadprog.
* Proximal spliting (a.k.a. ISTA) gradient descent for non smooth optimization.
* Spectral Projected Gradient solvers (spectral is optionnal but strongly recommended).
* Conditional gradient solver.

Planned integration are:

* L-BFGS for smooth optimization (interface to scipy and others)
* Stochastic gradients




## Installation

The library has been tested on Linux, MacOSX and Windows. 

The following Python modules are necessary and intalled automaticaly with:

- Numpy (>=1.11)
- Scipy (>=1.0)
- Cvxopt

The following modules are optional but obviously necessary to the solvers 
using them:

- Quadprog
- Gurobipy (official Python interface of GUROBI)
- Stdgrb

You can install the toolbox  by downloading it and then running:
```
python setup.py install --user # for user install (no root)
```

## Contributors

The main contributors of this module are:

- [Rémi Flamary](http://remi.flamary.com/)
- [Stéphane Canu](http://asi.insa-rouen.fr/enseignants/~scanu/)

