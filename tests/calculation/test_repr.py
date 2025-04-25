# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import importlib
from pathlib import PosixPath, WindowsPath  # these are required for the eval operation

from py4vasp import _calculation, calculation
from py4vasp._util import convert


def test_repr():
    for quantity in _calculation.QUANTITIES:
        instance = getattr(calculation, quantity)
        check_repr_is_consistent(instance, quantity)
    for group, quantities in _calculation.GROUPS.items():
        namespace = getattr(calculation, group)
        for quantity in quantities:
            instance = getattr(namespace, quantity)
            check_repr_is_consistent(instance, f"{group}_{quantity}")


def check_repr_is_consistent(instance, quantity):
    class_name = convert.to_camelcase(quantity)
    module = importlib.import_module(f"py4vasp._calculation.{quantity}")
    locals()[class_name] = getattr(module, class_name)
    copy = eval(repr(instance))
    assert copy.__class__ == instance.__class__
