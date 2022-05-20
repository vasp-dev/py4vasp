# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp.raw import *
from py4vasp.data import *
from py4vasp._util.convert import to_snakecase
from numpy import array
import pytest


@pytest.mark.xfail  # TODO: Adjust this for new interface
def test_repr(raw_data):
    tests = {
        Band: "multiple",
        Density: "Fe3O4 collinear",
        DielectricFunction: "electron",
        DielectricTensor: "dft",
        Dos: "Fe3O4",
        ElasticModulus: None,
        Energy: None,
        ForceConstant: "Sr2TiO4",
        Force: "Sr2TiO4",
        InternalStrain: "Sr2TiO4",
        Kpoint: "line",
        Magnetism: "collinear",
        PiezoelectricTensor: None,
        Polarization: None,
        Projector: "Fe3O4",
        Stress: "Sr2TiO4",
        Structure: "Fe3O4",
        Topology: "Fe3O4",
    }
    for class_, parameter in tests.items():
        raw = getattr(raw_data, to_snakecase(class_.__name__))(parameter)
        instance = class_(raw)
        copy = eval(repr(instance))
        assert copy.__class__ == class_
