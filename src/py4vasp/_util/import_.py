# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import importlib

from py4vasp import exception


class _LazyModule:
    """Defer importing an optional dependency until it is first used.

    The wrapped package is imported on the first attribute access rather than when
    :func:`optional` is called. This keeps ``import py4vasp`` fast: heavy optional
    dependencies (plotly, scipy, ase, ...) only load once code actually touches
    them. If the package is not installed, accessing an attribute raises
    :class:`~py4vasp.exception.ModuleNotInstalled`.
    """

    def __init__(self, name):
        self._name = name
        self._module = None

    def _resolve(self):
        if self._module is None:
            self._module = importlib.import_module(self._name)
        return self._module

    def __getattr__(self, attr):
        try:
            module = self._resolve()
        except Exception:
            raise exception.ModuleNotInstalled(
                "You use an optional part of py4vasp that relies on the package "
                f"'{self._name}'. Please install the package to use this functionality."
            ) from None
        return getattr(module, attr)


def optional(name):
    return _LazyModule(name)


def is_imported(module):
    try:
        module._resolve()
    except Exception:
        return False
    return True
