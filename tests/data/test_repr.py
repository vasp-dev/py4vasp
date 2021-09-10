from py4vasp.data import *
from py4vasp.raw import *
from numpy import array


def test_repr(raw_data):
    tests = {
        Band: "multiple",
        Density: "Fe3O4 collinear",
        Dos: "Fe3O4",
        Energy: None,
        Kpoints: "line",
        Magnetism: "collinear",
        Projectors: "Fe3O4",
        Structure: "Fe3O4 collinear",
        Topology: "Fe3O4",
        Trajectory: "Sr2TiO4",
    }
    for class_, parameter in tests.items():
        raw = getattr(raw_data, class_.__name__.lower())(parameter)
        instance = class_(raw)
        copy = eval(repr(instance))
        assert copy.__class__ == class_
