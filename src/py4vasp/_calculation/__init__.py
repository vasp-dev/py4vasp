# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy
import importlib
import pathlib

from py4vasp import exception
from py4vasp._util import convert

INPUT_FILES = ("INCAR", "KPOINTS", "POSCAR")
QUANTITIES = (
    "band",
    "bandgap",
    "born_effective_charge",
    "current_density",
    "density",
    "dielectric_function",
    "dielectric_tensor",
    "dos",
    "elastic_modulus",
    "electronic_minimization",
    "energy",
    "force",
    "force_constant",
    "internal_strain",
    "kpoint",
    "magnetism",
    "nics",
    "pair_correlation",
    "partial_density",
    "piezoelectric_tensor",
    "polarization",
    "potential",
    "projector",
    "stress",
    "structure",
    "system",
    "velocity",
    "workfunction",
    "_CONTCAR",
    "_dispersion",
    "_stoichiometry",
)
GROUPS = {
    "exciton": ("density", "eigenvector"),
    "phonon": ("band", "dos", "mode"),
}

__all__ = QUANTITIES


class Calculation:
    """Provide refinement functions for a the raw data of a VASP calculation run in any directory.

    The :data:`~py4vasp.calculation` object always reads the VASP calculation from the current
    working directory. This class gives you a more fine grained control so that you
    can use a Python script or Jupyter notebook in a different folder or rename the
    files that VASP produces.

    To create a new instance, you should use the classmethod :meth:`from_path` or
    :meth:`from_file` and *not* the constructor. This will ensure that the path to
    your VASP calculation is properly set and all features work as intended.
    The two methods allow you to read VASP results from a specific folder other
    than the working directory or a nondefault file name.

    With the Calculation instance, you can access the quantities VASP computes via
    the attributes of the object. The attributes are the same provided by the
    :data:`~py4vasp.calculation` object. You can find links to how to use these quantities
    below.

    Examples
    --------

    >>> calc = Calculation.from_path("path_to_your_calculation")
    >>> calc.dos.plot()         # to plot the density of states
    >>> calc.magnetism.read()   # to read the magnetic moments
    >>> calc.structure.print()  # to print the structure in a POSCAR format
    """

    def __init__(self, *args, **kwargs):
        if not kwargs.get("_internal"):
            message = """\
Please setup new Calculation instances using the classmethod Calculation.from_path()
instead of the constructor Calculation()."""
            raise exception.IncorrectUsage(message)

    @classmethod
    def from_path(cls, path_name):
        """Set up a Calculation for a particular path and so that all files are opened there.

        Parameters
        ----------
        path_name : str or pathlib.Path
            Name of the path associated with the calculation.

        Returns
        -------
        Calculation
            A calculation associated with the given path.
        """
        calc = cls(_internal=True)
        calc._path = pathlib.Path(path_name).expanduser().resolve()
        calc._file = None
        return calc

    @classmethod
    def from_file(cls, file_name):
        """Set up a Calculation from a particular file.

        Typically this limits the amount of information, you have access to, so prefer
        creating the instance with the :meth:`from_path` if possible.

        Parameters
        ----------
        file_name : str of pathlib.Path
            Name of the file from which the data is read.

        Returns
        -------
        Calculation
            A calculation accessing the data in the given file.
        """
        calc = cls(_internal=True)
        calc._path = pathlib.Path(file_name).expanduser().resolve().parent
        calc._file = file_name
        return calc

    def path(self):
        "Return the path in which the calculation is run."
        return self._path

    # Input files are not in current release
    # @property
    # def INCAR(self):
    #     "The INCAR file of the VASP calculation."
    #     return self._INCAR
    #
    # @INCAR.setter
    # def INCAR(self, incar):
    #     self._INCAR.write(str(incar))
    #
    # @property
    # def KPOINTS(self):
    #     "The KPOINTS file of the VASP calculation."
    #     return self._KPOINTS
    #
    # @KPOINTS.setter
    # def KPOINTS(self, kpoints):
    #     self._KPOINTS.write(str(kpoints))
    #
    # @property
    # def POSCAR(self):
    #     "The POSCAR file of the VASP calculation."
    #     return self._POSCAR
    #
    # @POSCAR.setter
    # def POSCAR(self, poscar):
    #     self._POSCAR.write(str(poscar))


def _add_all_refinement_classes(calc):
    for quantity in QUANTITIES:
        setattr(calc, quantity, _make_property(quantity))
    for group, quantities in GROUPS.items():
        setattr(calc, group, _make_group(group, quantities))
    return calc


def _make_property(quantity):
    # Be careful when refactoring this code, if the class_ is in a scope where it gets
    # changed like in the _add_all_refinement_classes routine, it may overwrite the
    # previous setting and all properties point to the same quantity.
    class_name = convert.to_camelcase(quantity)
    module = importlib.import_module(f"py4vasp._calculation.{quantity}")
    class_ = getattr(module, class_name)

    def get_quantity(self):
        if self._file is None:
            return class_.from_path(self._path)
        else:
            return class_.from_file(self._file)

    return property(get_quantity, doc=class_.__doc__)


def _make_group(group_name, quantities):
    # The Group class for each group is constructed on the fly. Alternatively you could
    # create a Group for every instance needed and construct the required properties as
    # needed. It is important that they are distinct classes.
    def get_group(self):
        class Group:
            def __init__(self, calculation):
                self._path = calculation._path
                self._file = calculation._file

        for quantity in quantities:
            full_name = f"{group_name}_{quantity}"
            setattr(Group, quantity, _make_property(full_name))
        return Group(self)

    return property(get_group)


Calculation = _add_all_refinement_classes(Calculation)


class DefaultCalculationFactory:
    """Provide refinement functions for a the raw data of a VASP calculation run in the
    current directory.

    Usually one is not directly interested in the raw data that is produced but
    wants to produce either a figure for a publication or some post-processing of
    the data. `calculation` contains multiple quantities that enable these kinds of
    workflows by extracting the relevant data from the HDF5 file and transforming
    them into an accessible format.

    Generally, all quantities provide a `read` function that extracts the data from the
    HDF5 file and puts it into a Python dictionary. Where it makes sense in addition
    a `plot` function is available that converts the data into a figure for Jupyter
    notebooks. In addition, data conversion routines `to_X` may be available
    transforming the data into another format or file, which may be useful to
    generate plots with tools other than Python. For the specifics, please refer to
    the documentation of the individual quantities.

    `calculation` reads the raw data from the current directory and from the default
    VASP output files. With the :class:`~py4vasp.Calculation` class, you can tailor
    the location of the files to your needs. Both have access to the same quantities,
    i.e., the two following examples are equivalent:

    .. rubric:: using :data:`~py4vasp.calculation` object

    >>> from py4vasp import calculation
    >>> calculation.dos.read()

    .. rubric:: using :class:`~py4vasp.Calculation` class

    >>> from py4vasp import Calculation
    >>> calc = Calculation.from_path(".")
    >>> calc.dos.read()

    In the latter example, you can change the path from which the data is extracted.
    """

    def __getattr__(self, attr):
        calc = Calculation.from_path(".")
        return getattr(calc, attr)

    def __setattr__(self, attr, value):
        calc = Calculation.from_path(".")
        return setattr(calc, attr, value)


# we use a factory instead of an instance of Calculation here so that changing the
# directory works -> calculation will always point to the current directory
calculation = DefaultCalculationFactory()
