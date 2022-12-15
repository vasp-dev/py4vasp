# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
""" Refine the raw data produced by VASP for plotting or analysis.

Usually one is not directly interested in the raw data that is produced but
wants to produce either a figure for a publication or some post-processing of
the data. This module contains multiple classes that enable these kinds of
workflows by extracting the relevant data from the HDF5 file and transforming
them into an accessible format. The classes also provide plotting functionality
to get a quick insight about the data, which can then be refined either within
python or a different tool to obtain publication-quality figures.

Generally, all classes provide a `read` function that extracts the data from the
HDF5 file and puts it into a Python dictionary. Where it makes sense in addition
a `plot` function is available that converts the data into a figure for Jupyter
notebooks. In addition, data conversion routines `to_X` may be available
transforming the data into another format or file, which may be useful to
generate plots with tools other than Python. For the specifics, please refer to
the documentation of the individual classes.
"""

from py4vasp._data.band import Band
from py4vasp._data.born_effective_charge import BornEffectiveCharge
from py4vasp._data.density import Density
from py4vasp._data.dielectric_function import DielectricFunction
from py4vasp._data.dielectric_tensor import DielectricTensor
from py4vasp._data.dispersion import Dispersion
from py4vasp._data.dos import Dos
from py4vasp._data.elastic_modulus import ElasticModulus
from py4vasp._data.energy import Energy
from py4vasp._data.fatband import Fatband
from py4vasp._data.force import Force
from py4vasp._data.force_constant import ForceConstant
from py4vasp._data.internal_strain import InternalStrain
from py4vasp._data.kpoint import Kpoint
from py4vasp._data.magnetism import Magnetism
from py4vasp._data.pair_correlation import PairCorrelation
from py4vasp._data.phonon_band import PhononBand
from py4vasp._data.phonon_dos import PhononDos
from py4vasp._data.piezoelectric_tensor import PiezoelectricTensor
from py4vasp._data.polarization import Polarization
from py4vasp._data.projector import Projector
from py4vasp._data.stress import Stress
from py4vasp._data.structure import Structure
from py4vasp._data.system import System
from py4vasp._data.topology import Topology
from py4vasp._data.velocity import Velocity
