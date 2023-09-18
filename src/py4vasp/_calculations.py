# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import inspect
import pathlib
from typing import Dict, List

from py4vasp import combine, exception
from py4vasp._util import convert


class Calculations:
    """A class to handle multiple Calculations all at once.

    This class combines the functionality of the Calculation class for more than one
    calculation. Create a Calculations object using either a wildcard for a set of
    paths or files or pass in paths and files directly. Then you can access the
    properties of all calculations via the attributes of the object.

    Examples
    --------
    >>> calcs = Calculations.from_paths(calc1="path_to_calc1", calc2="path_to_calc2")
    >>> calcs.energies.read() # returns a dictionary with the energies of calc1 and calc2
    >>> calcs.forces.read()   # returns a dictionary with the forces of calc1 and calc2
    >>> calcs.stresses.read() # returns a dictionary with the stresses of calc1 and calc2

    Notes
    -----
    To create new instances, you should use the classmethod :meth:`from_paths` or
    :meth:`from_files`. This will ensure that the paths to your VASP calculations are
    properly set and all features work as intended. Note that this is an alpha version
    and the API might change in the future.
    """

    def __init__(self, *args, **kwargs):
        if not kwargs.get("_internal"):
            message = """\
Please setup new CompareCalculations instance using the classmethod CompareCalculations.from_paths()
or CompareCalculations.from_files() instead of the constructor CompareCalculations()."""
            raise exception.IncorrectUsage(message)

    def _path_finder(**kwargs):
        for key, value in kwargs.items():
            if not (isinstance(value, pathlib.Path) or isinstance(value, str)):
                message = """\
Please provide a path to a VASP calculation as a string or pathlib.Path."""
                raise exception.IncorrectUsage(message)
            paths = pathlib.Path(value).expanduser().absolute()
            if "*" in paths.as_posix():
                paths = sorted(list(paths.parent.glob(paths.name)))
            else:
                paths = [paths]
            yield key, paths

    @classmethod
    def from_paths(cls, **kwargs):
        """Set up a Calculations object for paths.

        Setup a calculation for paths by passing in a dictionary with the name of the
        calculation as key and the path to the calculation as value.

        Parameters
        ----------
        **kwargs : Dict[str, str or pathlib.Path]
            A dictionary with the name of the calculation as key and the path to the
            calculation as value. Wildcards are allowed.
        """
        calculations = cls(_internal=True)
        calculations._paths = {}
        for key, paths in cls._path_finder(**kwargs):
            calculations._paths[key] = paths
        calculations = _add_all_combination_classes(
            calculations, _add_attribute_from_path
        )
        return calculations

    @classmethod
    def from_files(cls, **kwargs):
        """Set up a Calculations object from files.

        Setup a calculation for files by passing in a dictionary with the name of the
        calculation as key and the path to the calculation as value. Note that this
        limits the amount of information, you have access to, so prefer creating the
        instance with the :meth:`from_paths` if possible.

        Parameters
        ----------
        **kwargs : Dict[str, str or pathlib.Path]
            A dictionary with the name of the calculation as key and the files to the
            calculation as value. Wildcards are allowed.
        """
        calculations = cls(_internal=True)
        calculations._paths = {}
        calculations._files = {}
        for key, paths in cls._path_finder(**kwargs):
            basedir_paths = [path.parent for path in paths]
            calculations._paths[key] = basedir_paths
            calculations._files[key] = paths
        calculations = _add_all_combination_classes(
            calculations, _add_attribute_from_file
        )
        return calculations

    def paths(self) -> Dict[str, List[pathlib.Path]]:
        """Return the paths of the calculations."""
        return self._paths

    def files(self) -> Dict[str, List[pathlib.Path]]:
        """Return the files of the calculations."""
        return self._files

    def number_of_calculations(self) -> Dict[str, int]:
        """Return the number of calculations for each calculation."""
        return {key: len(value) for key, value in self._paths.items()}


def _add_attribute_from_path(calc, class_):
    instance = class_.from_paths(calc.paths())
    setattr(calc, convert.quantity_name(class_.__name__), instance)
    return calc


def _add_attribute_from_file(calc, class_):
    instance = class_.from_files(calc.files())
    setattr(calc, convert.quantity_name(class_.__name__), instance)
    return calc


def _add_all_combination_classes(calc, add_single_class):
    for _, class_ in inspect.getmembers(combine, inspect.isclass):
        calc = add_single_class(calc, class_)
    return calc
