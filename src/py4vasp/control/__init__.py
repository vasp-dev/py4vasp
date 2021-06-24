"""Setup input data for VASP calculations.

VASP requires several input files to execute. We provide some simple helper classes and
routines to generate these files from python. You can also use the routines to extract
the input files from a path.
"""
from .incar import INCAR
from .kpoints import KPOINTS
from .poscar import POSCAR
import inspect
import sys

_this_mod = sys.modules[__name__]
__all__ = [name for name, _ in inspect.getmembers(_this_mod, inspect.isclass)]
