# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pathlib
from typing import Dict

from py4vasp import Calculation, exception


class CompareCalculations:
    def __init__(self):
        pass

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
        compare = cls()
        compare._paths = {}
        for key, paths in cls._path_finder(**kwargs):
            compare._paths[key] = paths
        return compare

    @classmethod
    def from_files(cls, **kwargs):
        compare = cls()
        compare._paths = {}
        for key, paths in cls._path_finder(**kwargs):
            basedir_paths = [path.parent for path in paths]
            compare._paths[key] = basedir_paths
        return compare

    def paths(self) -> Dict[str, pathlib.Path]:
        return self._paths
