# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import types

import pytest

from py4vasp import calculation


@pytest.fixture
def OSZICAR(raw_data):
    raw_oszicar = raw_data.OSZICAR()
    oszicar = calculation.OSZICAR.from_data(raw_oszicar)
    oszicar.ref = types.SimpleNamespace()
    convergence_data = raw_oszicar.convergence_data
    oszicar.ref.iteration_number = convergence_data[:, 0]
    oszicar.ref.free_energy = convergence_data[:, 1]
    oszicar.ref.free_energy_change = convergence_data[:, 2]
    oszicar.ref.bandstructure_energy_change = convergence_data[:, 3]
    oszicar.ref.number_hamiltonian_evaluations = convergence_data[:, 4]
    oszicar.ref.norm_residual = convergence_data[:, 5]
    oszicar.ref.difference_charge_density = convergence_data[:, 6]
    return oszicar


def test_read(OSZICAR, Assert):
    actual = OSZICAR.read()
    expected = OSZICAR.ref
    Assert.allclose(actual["iteration_number"], expected.iteration_number)
    Assert.allclose(actual["free_energy"], expected.free_energy)
    Assert.allclose(actual["free_energy_change"], expected.free_energy_change)
    Assert.allclose(
        actual["bandstructure_energy_change"], expected.bandstructure_energy_change
    )
    Assert.allclose(
        actual["number_hamiltonian_evaluations"],
        expected.number_hamiltonian_evaluations,
    )
    Assert.allclose(actual["norm_residual"], expected.norm_residual)
    Assert.allclose(
        actual["difference_charge_density"], expected.difference_charge_density
    )


def test_plot(OSZICAR, Assert):
    graph = OSZICAR.plot()
    assert graph.xlabel == "Iteration number"
    assert graph.ylabel == "Free energy [eV]"
    assert len(graph.series) == 1
    Assert.allclose(graph.series[0].x, OSZICAR.ref.iteration_number)
    Assert.allclose(graph.series[0].y, OSZICAR.ref.free_energy)
