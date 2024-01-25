# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types
from unittest.mock import patch

import pytest

from py4vasp import data
from py4vasp._third_party.viewer import viewer3d

REF_Sr2TiO4 = """\
Sr2TiO4
6.9229000000000003
   1.0000000000000000    0.0000000000000000    0.0000000000000000
   0.6781122097386930    0.7349583872510080    0.0000000000000000
  -0.8390553410420490   -0.3674788590908430    0.4011800378743010
Sr Ti O
2 1 4
Direct
   0.6452900000000000    0.6452900000000000    0.0000000000000000
   0.3547100000000000    0.3547100000000000    0.0000000000000000
   0.0000000000000000    0.0000000000000000    0.0000000000000000
   0.8417800000000000    0.8417800000000000    0.0000000000000000
   0.1582300000000000    0.1582300000000000    0.0000000000000000
   0.5000000000000000    0.0000000000000000    0.5000000000000000
   0.0000000000000000    0.5000000000000000    0.5000000000000000"""

REF_Fe3O4 = """\
Fe3O4
1.0000000000000000
   5.1941269999999999    0.0000000000000000    0.0000000000000000
   0.0000000000000000    3.0893880000000000    0.0000000000000000
  -1.3770129362480001    0.0000000000000000    5.0950563617919995
Fe O
3 4
Selective dynamics
Direct
   0.0100000000000000    0.0100000000000000    0.0100000000000000  T F T
   0.5100000000000000    0.0100000000000000    0.5100000000000000  F T F
   0.0100000000000000    0.5100000000000000    0.5100000000000000  T F T
   0.7974500000000000    0.0100000000000000    0.2915200000000000  F T F
   0.2731000000000000    0.5100000000000000    0.2861100000000000  T F T
   0.2225500000000000    0.0100000000000000    0.7284800000000000  F T F
   0.7469000000000000    0.5100000000000000    0.7338900000000000  T F T
Lattice velocities and vectors
  2.39789553e+00 -3.00000000e-01 -3.00000000e-01
 -3.00000000e-01  6.54431821e-01 -3.00000000e-01
 -1.10383537e-01 -3.00000000e-01  2.29595993e+00
  5.19412700e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  3.08938800e+00  0.00000000e+00
 -1.37701294e+00  0.00000000e+00  5.09505636e+00
Cartesian
  0.00000000e+00  1.00000000e+00  1.41421356e+00
  1.73205081e+00  2.00000000e+00  2.23606798e+00
  2.44948974e+00  2.64575131e+00  2.82842712e+00
  3.00000000e+00  3.16227766e+00  3.31662479e+00
  3.46410162e+00  3.60555128e+00  3.74165739e+00
  3.87298335e+00  4.00000000e+00  4.12310563e+00
  4.24264069e+00  4.35889894e+00  4.47213595e+00"""


@pytest.fixture(params=["Sr2TiO4", "Fe3O4"])
def CONTCAR(raw_data, request):
    selection = request.param
    raw_contcar = raw_data.CONTCAR(selection)
    contcar = data.CONTCAR.from_data(raw_contcar)
    contcar.ref = types.SimpleNamespace()
    contcar.ref.structure = data.Structure.from_data(raw_data.structure(selection))[-1]
    contcar.ref.system = selection
    contcar.ref.selective_dynamics = raw_contcar.selective_dynamics
    contcar.ref.lattice_velocities = raw_contcar.lattice_velocities
    contcar.ref.ion_velocities = raw_contcar.ion_velocities
    contcar.ref.string = REF_Sr2TiO4 if selection == "Sr2TiO4" else REF_Fe3O4
    return contcar


class OptionalOutputCheck:
    def __init__(self, dict_, Assert):
        self.dict_ = dict_
        self.Assert = Assert

    def element_agrees(self, key, reference):
        if reference.is_none():
            assert key not in self.dict_
        else:
            self.Assert.allclose(self.dict_[key], reference)


def test_read(CONTCAR, Assert):
    actual = CONTCAR.read()
    expected = CONTCAR.ref.structure.read()
    Assert.allclose(actual["lattice_vectors"], expected["lattice_vectors"])
    Assert.allclose(actual["positions"], expected["positions"])
    assert actual["elements"] == expected["elements"]
    assert actual["names"] == expected["names"]
    assert actual["system"] == CONTCAR.ref.system
    check = OptionalOutputCheck(actual, Assert)
    check.element_agrees("selective_dynamics", CONTCAR.ref.selective_dynamics)
    check.element_agrees("lattice_velocities", CONTCAR.ref.lattice_velocities)
    check.element_agrees("ion_velocities", CONTCAR.ref.ion_velocities)


def test_plot(CONTCAR, not_core):
    obj = viewer3d.Viewer3d
    cm_init = patch.object(obj, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(obj, "show_cell")
    with cm_init as init, cm_cell as cell:
        fig = CONTCAR.plot()
        init.assert_called_once()
        cell.assert_called_once()


def test_print(CONTCAR, format_):
    actual, _ = format_(CONTCAR)
    assert actual["text/plain"] == CONTCAR.ref.string
    assert actual == {"text/plain": CONTCAR.ref.string}


def test_factory_methods(raw_data, check_factory_methods):
    raw_contcar = raw_data.CONTCAR("Sr2TiO4")
    check_factory_methods(data.CONTCAR, raw_contcar)
