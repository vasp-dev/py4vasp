import importlib

from py4vasp import exception


class _ModulePlaceholder:
    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        raise exception.ModuleNotInstalled(
            "You use an optional part of py4vasp that relies on the package "
            f"'{self._name}'. Please install the package to use this functionality."
        )


def optional(name):
    try:
        return importlib.import_module(name)
    except:
        return _ModulePlaceholder(name)
