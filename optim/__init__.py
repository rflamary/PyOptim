""" PyOptim Python numerical optimization toolbox"""


from . import utils
from . import solvers
from . import loss
from . import prox
from . import proj

__version__="0.1.0"

from .solvers import fmin_prox,fmin_proj,fmin_conj


__all__ = ["utils","solvers","loss","prox",'proj',
           "fmin_prox","fmin_proj","fmin_conj"]

