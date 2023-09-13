# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import importlib
import inspect
import pathlib
from typing import Dict, List

from py4vasp import data, exception


def match_combine_with_refinement(combine_name: str):
    combine_to_refinement_name = {
        "Energies": "Energy",
        "Forces": "Force",
        "Stresses": "Stress",
    }
    for _, class_ in inspect.getmembers(data, inspect.isclass):
        if class_.__name__ == combine_to_refinement_name[combine_name]:
            return class_
    else:
        raise exception.IncorrectUsage(
            f"Could not find refinement class for {combine_name}."
        )


class BaseCombine:
    def __init__(self):
        pass

    @classmethod
    def from_paths(cls, paths: Dict[str, List[pathlib.Path]]):
        base = cls()
        refinement = match_combine_with_refinement(cls.__name__)
        setattr(base, f"_{cls.__name__.lower()}", {})
        for key, path in paths.items():
            all_refinements = [refinement.from_path(_path) for _path in path]
            base.__getattribute__(f"_{cls.__name__.lower()}")[key] = all_refinements
        return base

    @classmethod
    def from_files(cls, files: Dict[str, List[pathlib.Path]]):
        base = cls()
        refinement = match_combine_with_refinement(cls.__name__)
        setattr(base, f"_{cls.__name__.lower()}", {})
        for key, file in files.items():
            all_refinements = [refinement.from_file(_file) for _file in file]
            base.__getattribute__(f"_{cls.__name__.lower()}")[key] = all_refinements
        return base

    def to_dict(self, *args, **kwargs):
        _data = {}
        _class_name = f"_{self.__class__.__name__.lower()}"
        keyval_refinements = self.__getattribute__(_class_name).items()
        for key, refinement in keyval_refinements:
            _data[key] = [
                _refinement.read(*args, **kwargs) for _refinement in refinement
            ]
        return _data

    def read(self, *args, **kwargs):
        return self.to_dict(*args, **kwargs)
