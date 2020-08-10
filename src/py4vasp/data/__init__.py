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

import plotly.io as pio
import cufflinks as cf
import inspect
import sys

pio.templates.default = "ggplot2"
cf.go_offline()
cf.set_config_file(theme="ggplot")

_this_mod = sys.modules[__name__]
__all__ = [name for name, _ in inspect.getmembers(_this_mod, inspect.isclass)]
