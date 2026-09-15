# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp._calculation.optics import Optics
from py4vasp._demo import showcase
from py4vasp._demo.showcase import dielectric_function, electronic_structure


@pytest.fixture
def raw_dielectric_function():
    return dielectric_function.electron()


@pytest.fixture
def epsilon(raw_dielectric_function):
    real, imaginary = np.moveaxis(
        np.array(raw_dielectric_function.dielectric_function), -1, 0
    )
    return real + 1j * imaginary


def test_dielectric_function_covers_the_optical_range(raw_dielectric_function):
    energies = np.array(raw_dielectric_function.energies)
    assert energies.shape == (showcase.NUMBER_POINTS,)
    assert energies[0] == 0
    # far enough beyond the gap that the whole absorption edge is on the axis
    assert energies[-1] > 3 * electronic_structure.BAND_GAP


def test_dielectric_function_is_passive(epsilon):
    # a material that amplifies light instead of absorbing it would have negative
    # imaginary part
    assert np.all(epsilon.imag >= 0)
    # and its static dielectric constant screens, so it exceeds that of vacuum
    assert np.all(np.diagonal(epsilon[:, :, 0]) > 1)


def test_absorption_switches_on_at_the_gap(raw_dielectric_function, epsilon):
    energies = np.array(raw_dielectric_function.energies)
    gap = electronic_structure.BAND_GAP
    below = np.argmin(np.abs(energies - 0.5 * gap))
    at_gap = np.argmin(np.abs(energies - gap))
    assert epsilon.imag[0, 0, at_gap] > 10 * epsilon.imag[0, 0, below]


def test_dielectric_function_is_tetragonal(epsilon, Assert):
    Assert.allclose(epsilon[0, 0], epsilon[1, 1])
    assert not np.allclose(epsilon[0, 0], epsilon[2, 2])
    off_diagonal = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]
    for row, column in off_diagonal:
        Assert.allclose(epsilon[row, column], np.zeros(showcase.NUMBER_POINTS))


def test_optics_conserves_the_incident_light(raw_dielectric_function):
    optics = Optics.from_data(raw_dielectric_function).read()
    total = optics["transmission"] + optics["absorption"] + optics["reflectivity"]
    # every photon is transmitted, absorbed or reflected wherever none of the three is
    # clipped at the boundary of its range
    transmitting = optics["transmission"] > 0
    assert np.allclose(total[transmitting], 1)


def test_optics_transmits_below_the_gap_and_absorbs_above(raw_dielectric_function):
    optics = Optics.from_data(raw_dielectric_function).read()
    energies = optics["energies"]
    gap = electronic_structure.BAND_GAP
    below = energies < 0.5 * gap
    above = (energies > 1.5 * gap) & (energies < 2.5 * gap)
    assert np.mean(optics["transmission"][below]) > 0.7
    assert np.mean(optics["absorption"][above]) > np.mean(optics["absorption"][below])
