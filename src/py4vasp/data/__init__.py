# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
""" Refine the raw data produced by VASP for plotting or analysis.

Usually one is not directly interested in the raw data that is produced, but
wants to produce either a figure for a publication or some post processing of
the data. This module contains multiple classes that enable these kind of
workflows by extracting the relevant data from the HDF5 file and transforming
them into an accessible format. The classes also provide plotting functionality
to get a quick insight about the data, which can then be refined either within
python or a different tool to obtain publication quality figures.

Generally, all classes provide a `read` function that extracts the data from the
HDF5 file and puts it into a Python dictionary. Where it makes sense in addition
a `plot` function is available that converts the data into a figure for Jupyter
notebooks. In addition, data conversion routines `to_X` may be available
transforming the data into another format or file, which may be useful to
generate plots with tools other than Python. For the specifics, please refer to
the documentation of the individual classes.
"""

from .band import Band
from .dos import Dos
from .dielectric_function import DielectricFunction
from .dielectric_tensor import DielectricTensor
from .elastic_modulus import ElasticModulus
from .energy import Energy
from .kpoint import Kpoint
from .piezoelectric_tensor import PiezoelectricTensor
from .polarization import Polarization
from .projector import Projector
from .topology import Topology
from .viewer3d import Viewer3d
from .structure import Structure
from .system import System
from .born_effective_charge import BornEffectiveCharge
from .density import Density
from .force import Force
from .force_constant import ForceConstant
from .internal_strain import InternalStrain
from .magnetism import Magnetism
from .stress import Stress

import plotly.io as pio
import plotly.graph_objects as go
import inspect
import sys

_this_mod = sys.modules[__name__]
__all__ = [name for name, _ in inspect.getmembers(_this_mod, inspect.isclass)]

pio.templates["vasp"] = go.layout.Template(
    layout={"colorway": ["#4C265F", "#2FB5AB", "#2C68FC", "#A82C35", "#808080"]}
)
pio.templates.default = "ggplot2+vasp"
