# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import inspect
import pathlib

from py4vasp import _data, calculation, control, data, exception
from py4vasp._util import convert


class Calculation:
    """Manage access to input and output of VASP calculations.

    This is the main user interface if you want to simply investigate the results of VASP
    calculations. Create a Calculation object associated with the VASP calculation that you
    run. Then you can access the properties of that calculation via the attributes of the
    object.

    Examples
    --------

    >>> calc = Calculation.from_path("path_to_your_calculation")
    >>> calc.dos.plot()         # to plot the density of states
    >>> calc.magnetism.read()   # to read the magnetic moments
    >>> calc.structure.print()  # to print the structure in a POSCAR format

    Notes
    -----
    To create new instances, you should use the classmethod :meth:`from_path` or
    :meth:`from_file`. This will ensure that the path to your VASP calculation is
    properly set and all features work as intended.

    .. autosummary::

       from_file
       from_path
       path
       INCAR
       KPOINTS
       POSCAR
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
        calc = _add_all_refinement_classes(calc, _add_attribute_from_path)
        return _add_input_files(calc)

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
        calc = _add_all_refinement_classes(calc, _AddAttributeFromFile(file_name))
        return _add_input_files(calc)

    def path(self):
        "Return the path in which the calculation is run."
        return self._path

    @property
    def INCAR(self):
        "The INCAR file of the VASP calculation."
        return self._INCAR

    @INCAR.setter
    def INCAR(self, incar):
        self._INCAR.write(str(incar))

    @property
    def KPOINTS(self):
        "The KPOINTS file of the VASP calculation."
        return self._KPOINTS

    @KPOINTS.setter
    def KPOINTS(self, kpoints):
        self._KPOINTS.write(str(kpoints))

    @property
    def POSCAR(self):
        "The POSCAR file of the VASP calculation."
        return self._POSCAR

    @POSCAR.setter
    def POSCAR(self, poscar):
        self._POSCAR.write(str(poscar))


def _add_all_refinement_classes(calc, add_single_class):
    for name in calculation.__all__:
        calc = add_single_class(calc, name)
    return calc


def _add_attribute_from_path(calc, name):
    class_ = getattr(data, convert.to_camelcase(name))
    instance = class_.from_path(calc.path())
    setattr(calc, name, instance)
    return calc


class _AddAttributeFromFile:
    def __init__(self, file_name):
        self._file_name = file_name

    def __call__(self, calc, name):
        class_ = getattr(data, convert.to_camelcase(name))
        instance = class_.from_file(self._file_name)
        setattr(calc, name, instance)
        return calc


def _add_to_documentation(calc, name):
    calc.__doc__ += f"   py4vasp.calculation.{name}\n    "
    return calc


Calculation = _add_all_refinement_classes(Calculation, _add_to_documentation)


def _add_input_files(calc):
    calc._INCAR = control.INCAR(calc.path())
    calc._KPOINTS = control.KPOINTS(calc.path())
    calc._POSCAR = control.POSCAR(calc.path())
    return calc
