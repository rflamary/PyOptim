""" PyOptim Python numerical optimization toolbox"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License


from . import utils
from . import solvers
from . import loss
from . import prox
from . import proj
from . import stdsolvers
from . import bench


__version__ = "0.1.0"

from .solvers import fmin_prox, fmin_proj, fmin_cond
from .stdsolvers import lp_solve, qp_solve

__all__ = ["utils", "solvers", 'stdsolvers', "loss", "prox", 'proj',
           "fmin_prox", "fmin_proj", "fmin_cond", 'bench', "lp_solve",
           "qp_solve"]
