# -*- coding: utf-8 -*-

# Has to be first
from . import _distributor_init

from .__version__ import __version__

from .core import GenSVM
from .gridsearch import GenSVMGridSearchCV
