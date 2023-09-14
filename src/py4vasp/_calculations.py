# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import inspect
import pathlib
from typing import Dict, List

from py4vasp import combine, exception
from py4vasp._util import convert


class Calculations:
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
            paths = pathlib.Path(value).expanduser().resolve()
            if "*" in paths.as_posix():
                paths = sorted(list(paths.parent.glob(paths.name)))
            else:
                paths = [paths]
            yield key, paths

    @classmethod
    def from_paths(cls, **kwargs):
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
        return self._paths

    def files(self) -> Dict[str, List[pathlib.Path]]:
        return self._files

    def number_of_calculations(self) -> Dict[str, int]:
        return {key: len(value) for key, value in self._paths.items()}


def _add_attribute_from_path(calc, class_):
    instance = class_.from_paths(calc.paths())
    setattr(calc, convert.to_snakecase(class_.__name__), instance)
    return calc


def _add_attribute_from_file(calc, class_):
    instance = class_.from_files(calc.files())
    setattr(calc, convert.to_snakecase(class_.__name__), instance)
    return calc


def _add_all_combination_classes(calc, add_single_class):
    for _, class_ in inspect.getmembers(combine, inspect.isclass):
        calc = add_single_class(calc, class_)
    return calc
