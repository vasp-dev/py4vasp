# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp import demo
from py4vasp._demo.showcase import elastic_modulus
from py4vasp._util.tensor import symmetry_reduce

XX, YY, ZZ, XY, YZ, ZX = range(6)  # the Voigt order py4vasp prints the modulus in


@pytest.fixture
def raw_elastic_modulus():
    return elastic_modulus.Sr2TiO4()


@pytest.fixture(params=["clamped_ion", "relaxed_ion"])
def tensor(raw_elastic_modulus, request):
    return np.array(getattr(raw_elastic_modulus, request.param))


def _voigt(tensor):
    return symmetry_reduce(symmetry_reduce(tensor).T).T


def test_tensor_couples_two_strains_of_three_directions(tensor):
    assert tensor.shape == (3, 3, 3, 3)


def test_tensor_has_the_symmetry_of_a_stiffness(tensor, Assert):
    # stress and strain are symmetric, and the stiffness is a second derivative of the
    # energy, so the two pairs of indices can be exchanged
    Assert.allclose(tensor, tensor.transpose(1, 0, 2, 3))
    Assert.allclose(tensor, tensor.transpose(0, 1, 3, 2))
    Assert.allclose(tensor, tensor.transpose(2, 3, 0, 1))


def test_tensor_has_the_symmetry_of_a_tetragonal_crystal(tensor, Assert):
    voigt = _voigt(tensor)
    assert voigt[XX, XX] == voigt[YY, YY]
    assert voigt[XX, ZZ] == voigt[YY, ZZ]
    assert voigt[YZ, YZ] == voigt[ZX, ZX]
    # a stretch does not shear the crystal and no shear couples to another one
    Assert.allclose(voigt[:3, 3:], np.zeros((3, 3)))
    shears = voigt[3:, 3:]
    Assert.allclose(shears, np.diag(np.diag(shears)))


def test_crystal_is_mechanically_stable(tensor):
    assert np.all(np.linalg.eigvalsh(_voigt(tensor)) > 0)


def test_relaxing_the_ions_softens_the_crystal(raw_elastic_modulus):
    clamped_ion = _voigt(np.array(raw_elastic_modulus.clamped_ion))
    relaxed_ion = _voigt(np.array(raw_elastic_modulus.relaxed_ion))
    softening = np.linalg.eigvalsh(clamped_ion - relaxed_ion)
    assert np.all(softening >= 0)
    assert np.any(softening > 0)


def test_bulk_modulus_is_the_one_of_an_oxide_in_kBar(tensor):
    # the Voigt average of an oxide perovskite is of the order of 150 GPa = 1500 kBar
    bulk_modulus = np.sum(_voigt(tensor)[:3, :3]) / 9
    assert 1000 < bulk_modulus < 2000


def test_calculation_contains_the_showcase(raw_elastic_modulus, tmp_path, Assert):
    calculation = demo.calculation(tmp_path / "elastic_modulus")
    actual = calculation.elastic_modulus.read()
    Assert.allclose(actual["clamped_ion"], raw_elastic_modulus.clamped_ion)
    Assert.allclose(actual["relaxed_ion"], raw_elastic_modulus.relaxed_ion)
