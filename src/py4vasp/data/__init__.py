""" Refine the raw data produced by Vasp for plotting or analysis.

Usually one is not directly interested in the raw data that is produced, but
wants to produce either a figure for a publication or some post processing of
the data. This module contains multiple classes that enable these kind of
workflows by extracting the relevant data from the HDF5 file and transforming
them into an accessible format. The classes also provide plotting functionality
to get a quick insight about the data, which can then be refined either within
python or a different tool to obtain publication quality figures.

Generally, all classes provide a `read` function that extracts the data from the
HDF5 file and puts it into a python dictionary. Where it makes sense in addition
a `plot` function is available that converts the data into a figure for Jupyter
notebooks. In addition, data conversion routines `to_X` may be available
transforming the data into another format or file, which may be useful to
generate plots with tools other than python. For the specifics, please refer to
the documentation of the individual classes.
"""

from .band import Band
from .dos import Dos
from .energy import Energy
from .kpoints import Kpoints
from .projectors import Projectors
from .topology import Topology
from .trajectory import Trajectory
from .viewer3d import Viewer3d
from .structure import Structure
from py4vasp.exceptions import NotImplementException

import plotly.io as pio
import cufflinks as cf
import inspect
import sys

pio.templates.default = "ggplot2"
cf.go_offline()
cf.set_config_file(theme="ggplot")

_this_mod = sys.modules[__name__]
_class_names = [name for name, _ in inspect.getmembers(_this_mod, inspect.isclass)]
_classes = [value for _, value in inspect.getmembers(_this_mod, inspect.isclass)]
_functions = set(("read", "plot"))
for c in _classes:
    for name, _ in inspect.getmembers(c, inspect.isfunction):
        if "to_" in name:
            _functions.add(name)
__all__ = _class_names + list(_functions)


def get_function_if_possible(obj, name):
    try:
        return getattr(obj, name)
    except AttributeError as err:
        class_ = obj.__class__.__name__
        msg = "For the {} no {} function is implemented.".format(class_, name)
        raise NotImplementException(msg) from err


def _wrapper_factory(module, name):
    def wrapper(cls, *args, **kwargs):
        with cls.from_file() as obj:
            function = get_function_if_possible(obj, name)
            return function(*args, **kwargs)

    setattr(module, name, wrapper)


for function in _functions:
    _wrapper_factory(_this_mod, function)
