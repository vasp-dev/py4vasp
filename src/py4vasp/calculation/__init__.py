# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Provide refinement functions for a the raw data of a VASP calculation run in the
current directory.

Usually one is not directly interested in the raw data that is produced but
wants to produce either a figure for a publication or some post-processing of
the data. This package contains multiple modules that enable these kinds of
workflows by extracting the relevant data from the HDF5 file and transforming
them into an accessible format. The modules also provide plotting functionality
to get a quick insight about the data, which can then be refined either within
python or a different tool to obtain publication-quality figures.

Generally, all modules provide a `read` function that extracts the data from the
HDF5 file and puts it into a Python dictionary. Where it makes sense in addition
a `plot` function is available that converts the data into a figure for Jupyter
notebooks. In addition, data conversion routines `to_X` may be available
transforming the data into another format or file, which may be useful to
generate plots with tools other than Python. For the specifics, please refer to
the documentation of the individual modules.

The raw data is read from the current directory. The :class:`~py4vasp.Calculation`
class provides a more flexible interface with which you can determine the source
directory or file for the VASP calculation manually. That class exposes functions
of the modules as methods of attributes, i.e., the two following examples are
equivalent:

.. rubric:: using :mod:`~py4vasp.calculation` module

>>> from py4vasp import calculation
>>> calculation.dos.read()

.. rubric:: using :class:`~py4vasp.Calculation` class

>>> from py4vasp import Calculation
>>> calc = Calculation.from_path(".")
>>> calc.dos.read()

In the latter example, you can change the path from which the data is extracted.
"""
import importlib

from py4vasp import exception
from py4vasp._util import convert

__all__ = (
    "band",
    "bandgap",
    "born_effective_charge",
    "CONTCAR",
    "density",
    "dielectric_function",
    "dielectric_tensor",
    "dos",
    "elastic_modulus",
    "energy",
    "fatband",
    "force",
    "force_constant",
    "internal_strain",
    "kpoint",
    "magnetism",
    "OSZICAR",
    "pair_correlation",
    "phonon_band",
    "phonon_dos",
    "piezoelectric_tensor",
    "polarization",
    "potential",
    "projector",
    "stress",
    "structure",
    "system",
    "topology",
    "velocity",
    "workfunction",
)
_private = ("dispersion",)


def __getattr__(attr):
    if attr in (__all__ + _private):
        module = importlib.import_module(f"py4vasp.calculation._{attr}")
        class_ = getattr(module, convert.to_camelcase(attr))
        return class_.from_path(".")
    else:
        message = f"Could not find {attr} in the possible attributes, please check the spelling"
        raise exception.MissingAttribute(message)
