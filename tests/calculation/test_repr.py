# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import importlib
from pathlib import PosixPath, WindowsPath

from py4vasp import _calculation, calculation
from py4vasp._util import convert


def test_repr():
    for name in _calculation.QUANTITIES:
        instance = getattr(calculation, name)
        class_name = convert.to_camelcase(name)
        module = importlib.import_module(f"py4vasp._calculation.{name}")
        locals()[class_name] = getattr(module, class_name)
        print(repr(instance))
        copy = eval(repr(instance))
        assert copy.__class__ == instance.__class__
