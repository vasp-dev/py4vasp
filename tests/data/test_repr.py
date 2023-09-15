# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp.data import *


def test_repr():
    tests = [
        Band,
        Bandgap,
        BornEffectiveCharge,
        CONTCAR,
        Density,
        DielectricFunction,
        DielectricTensor,
        Dos,
        ElasticModulus,
        Energy,
        Fatband,
        ForceConstant,
        Force,
        InternalStrain,
        Kpoint,
        Magnetism,
        PairCorrelation,
        PhononBand,
        PhononDos,
        PiezoelectricTensor,
        Polarization,
        Projector,
        Stress,
        Structure,
        System,
        Topology,
    ]
    for class_ in tests:
        instance = class_.from_data("mock_data")
        copy = eval(repr(instance))
        assert copy.__class__ == class_
