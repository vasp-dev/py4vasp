# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import importlib

from py4vasp import calculation
from py4vasp._util import convert


def test_repr():
    for name in calculation.__all__:
        instance = getattr(calculation, name)
        class_name = convert.to_camelcase(name)
        module = importlib.import_module(f"py4vasp.calculation._{name}")
        locals()[class_name] = getattr(module, class_name)
        copy = eval(repr(instance))
        assert copy.__class__ == instance.__class__
