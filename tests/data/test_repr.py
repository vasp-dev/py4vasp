# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import calculation
from py4vasp.calculation.data_all import *


def test_repr():
    for name in calculation.__all__:
        instance = getattr(calculation, name)
        copy = eval(repr(instance))
        assert copy.__class__ == instance.__class__
