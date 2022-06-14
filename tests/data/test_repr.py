# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp.data import *


def test_repr():
    tests = [
        Band,
        Density,
        DielectricFunction,
        DielectricTensor,
        Dos,
        ElasticModulus,
        Energy,
        ForceConstant,
        Force,
        InternalStrain,
        Kpoint,
        Magnetism,
        PiezoelectricTensor,
        Polarization,
        Projector,
        Stress,
        Structure,
        Topology,
    ]
    for class_ in tests:
        instance = class_.from_data("mock_data")
        copy = eval(repr(instance))
        assert copy.__class__ == class_
