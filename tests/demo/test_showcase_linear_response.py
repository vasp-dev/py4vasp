# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp import demo
from py4vasp._demo.showcase import linear_response, structure


@pytest.fixture
def charge_tensors():
    return np.array(linear_response.born_effective_charge_Sr2TiO4().charge_tensors)


@pytest.fixture
def raw_tensor():
    return linear_response.dielectric_tensor_Sr2TiO4()


@pytest.mark.parametrize("component", ("electron", "ion", "independent_particle"))
def test_dielectric_tensor_is_symmetric_and_tetragonal(raw_tensor, component, Assert):
    tensor = np.array(getattr(raw_tensor, component))
    Assert.allclose(tensor, np.diag(np.diag(tensor)))
    assert tensor[0, 0] == tensor[1, 1] != tensor[2, 2]


def test_ions_screen_the_field_further(raw_tensor):
    # the relaxed-ion tensor adds the ionic contribution to the clamped-ion one, so a
    # stable crystal can only become more polarizable
    assert np.all(np.diag(np.array(raw_tensor.ion)) > 0)


def test_born_charges_cover_every_atom(charge_tensors):
    _, number_atoms, _ = np.array(structure.Sr2TiO4().positions).shape
    assert charge_tensors.shape == (number_atoms, 3, 3)


def test_born_charges_obey_the_acoustic_sum_rule(charge_tensors, Assert):
    # shifting the whole crystal must not produce a net force in a field
    Assert.allclose(charge_tensors.sum(axis=0), np.zeros((3, 3)))


def test_equatorial_oxygen_is_anomalous_along_its_bond(charge_tensors):
    # the large charge of an oxygen in the TiO2 plane points along its Ti-O bond:
    # atom 6 sits along y and atom 7 along x of the titanium at the origin
    assert np.argmin(np.diag(charge_tensors[5])) == 1
    assert np.argmin(np.diag(charge_tensors[6])) == 0


def test_demo_calculation_contains_linear_response(tmp_path, Assert):
    calculation = demo.calculation(tmp_path / "calculation")
    Assert.allclose(
        calculation.born_effective_charge.read()["charge_tensors"],
        linear_response.born_effective_charge_Sr2TiO4().charge_tensors,
    )
    Assert.allclose(
        calculation.dielectric_tensor.read()["clamped_ion"],
        linear_response.dielectric_tensor_Sr2TiO4().electron,
    )
