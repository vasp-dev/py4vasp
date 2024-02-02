# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import inspect
import pathlib
from typing import Dict, List

from py4vasp import calculation, exception


def _match_combine_with_refinement(combine_name: str):
    combine_to_refinement_name = {
        "Energies": "energy",
        "Forces": "force",
        "Stresses": "stress",
    }
    return getattr(calculation, combine_to_refinement_name[combine_name])
    # for _, class_ in inspect.getmembers(data_depr, inspect.isclass):
    #     if class_.__name__ == combine_to_refinement_name[combine_name]:
    #         return class_
    # else:
    #     raise exception.IncorrectUsage(
    #         f"Could not find refinement class for {combine_name}."
    #     )


class BaseCombine:
    """A class to handle multiple refinements all at once.

    This class combines the functionality of the refinement class for more than one
    refinement. Create a BaseCombine object using either a wildcard for a set of
    paths or files or pass in paths and files directly. Then you can access the
    properties of all refinements via the attributes of the object.

    Notes
    -----
    To create new instances, you should use the classmethod :meth:`from_paths` or
    :meth:`from_files`. This will ensure that the paths to your VASP calculations are
    properly set and all features work as intended. Note that this is an alpha version
    and the API might change in the future.
    """

    def __init__(self):
        pass

    @classmethod
    def from_paths(cls, paths: Dict[str, List[pathlib.Path]]):
        """Set up a BaseCombine object for paths.

        Setup the object for paths by passing in a dictionary with the name of the
        calculation as key and the path to the calculation as value.

        Parameters
        ----------
        paths : Dict[str, List[pathlib.Path]]
            A dictionary with the name of the calculation as key and the path to the
            calculation as value.
        """
        base = cls()
        refinement = _match_combine_with_refinement(cls.__name__)
        setattr(base, f"_{cls.__name__.lower()}", {})
        for key, path in paths.items():
            all_refinements = [refinement.from_path(_path) for _path in path]
            base.__getattribute__(f"_{cls.__name__.lower()}")[key] = all_refinements
        return base

    @classmethod
    def from_files(cls, files: Dict[str, List[pathlib.Path]]):
        """Set up a BaseCombine object for files.

        Setup the object for files by passing in a dictionary with the name of the
        calculation as key and the path to the calculation as value.

        Parameters
        ----------
        files : Dict[str, List[pathlib.Path]]
            A dictionary with the name of the calculation as key and the path to the
            calculation as value.
        """
        base = cls()
        refinement = _match_combine_with_refinement(cls.__name__)
        setattr(base, f"_{cls.__name__.lower()}", {})
        for key, file in files.items():
            all_refinements = [refinement.from_file(_file) for _file in file]
            base.__getattribute__(f"_{cls.__name__.lower()}")[key] = all_refinements
        return base

    def _to_dict(self, *args, **kwargs):
        _data = {}
        _class_name = f"_{self.__class__.__name__.lower()}"
        keyval_refinements = self.__getattribute__(_class_name).items()
        for key, refinement in keyval_refinements:
            _data[key] = [
                _refinement.read(*args, **kwargs) for _refinement in refinement
            ]
        return _data

    def read(self, *args, **kwargs):
        """Read the data from the :meth:`read` method of the refinement class."""
        return self._to_dict(*args, **kwargs)
