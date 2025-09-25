# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp._calculation.electron_phonon_chemical_potential import (
    ElectronPhononChemicalPotential as ChemicalPotential,
)


@pytest.fixture(params=["carrier_den", "mu", "carrier_per_cell"])
def mu_tag(request):
    return request.param


@pytest.fixture
def chemical_potential(raw_data, mu_tag):
    raw_chemical_potential = raw_data.electron_phonon_chemical_potential(mu_tag)
    chemical_potential = ChemicalPotential.from_data(raw_chemical_potential)
    chemical_potential.ref = types.SimpleNamespace()
    chemical_potential.ref.fermi_energy = raw_chemical_potential.fermi_energy
    chemical_potential.ref.chemical_potential = (
        raw_chemical_potential.chemical_potential[:]
    )
    chemical_potential.ref.carrier_density = raw_chemical_potential.carrier_density[:]
    chemical_potential.ref.temperatures = raw_chemical_potential.temperatures[:]
    chemical_potential.ref.mu_tag = mu_tag
    if mu_tag == "carrier_den":
        chemical_potential.ref.mu_val = raw_chemical_potential.carrier_den[:]
    elif mu_tag == "mu":
        chemical_potential.ref.mu_val = raw_chemical_potential.mu[:]
    else:  # mu_tag == "carrier_per_cell"
        chemical_potential.ref.mu_val = raw_chemical_potential.carrier_per_cell[:]
    return chemical_potential


def test_read(chemical_potential, Assert):
    actual = chemical_potential.read()
    mu_tag = f"selfen_{chemical_potential.ref.mu_tag}"
    expected_keys = {
        "fermi_energy",
        "chemical_potential",
        "carrier_density",
        "temperatures",
        mu_tag,
    }
    assert actual.keys() == expected_keys
    Assert.allclose(actual["fermi_energy"], chemical_potential.ref.fermi_energy)
    Assert.allclose(
        actual["chemical_potential"], chemical_potential.ref.chemical_potential
    )
    Assert.allclose(actual["carrier_density"], chemical_potential.ref.carrier_density)
    Assert.allclose(actual["temperatures"], chemical_potential.ref.temperatures)
    Assert.allclose(actual[mu_tag], chemical_potential.ref.mu_val)
