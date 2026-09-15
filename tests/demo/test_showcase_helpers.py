# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp import _demo
from py4vasp._demo import showcase


@pytest.fixture
def energies():
    return np.linspace(-8, 8, showcase.NUMBER_POINTS)


def test_showcase_resolves_finer_than_the_test_data():
    # the producers in _demo are sized for fast tests; presentation data needs enough
    # points that a curve reads as a curve and a relaxation as a relaxation
    assert showcase.NUMBER_POINTS > _demo.NUMBER_POINTS
    assert showcase.NUMBER_STEPS > _demo.NUMBER_STEPS


def test_broaden_peaks_at_the_level(energies):
    level = 1.5
    dos = showcase.broaden(energies, [level])
    spacing = energies[1] - energies[0]
    assert dos.shape == energies.shape
    assert np.all(dos >= 0)
    assert abs(energies[np.argmax(dos)] - level) < spacing


def test_broaden_integrates_to_the_total_weight(energies):
    weights = [1.0, 2.0, 0.5]
    dos = showcase.broaden(energies, levels=[-3.0, 0.0, 2.5], weights=weights)
    assert np.trapezoid(dos, energies) == pytest.approx(sum(weights), rel=1e-6)


def test_broaden_scales_with_the_weights(energies, Assert):
    levels = [-1.0, 1.0]
    Assert.allclose(
        showcase.broaden(energies, levels, weights=2.0),
        2 * showcase.broaden(energies, levels),
    )


def test_broaden_flattens_multidimensional_levels(energies, Assert):
    # eigenvalues arrive shaped (kpoint, band) and the weights follow that shape
    levels = np.linspace(-2, 2, 12).reshape(4, 3)
    Assert.allclose(
        showcase.broaden(energies, levels, weights=np.ones_like(levels)),
        showcase.broaden(energies, levels.ravel()),
    )


def test_broaden_width_controls_the_peak_height(energies):
    sharp = showcase.broaden(energies, [0.0], width=0.05)
    broad = showcase.broaden(energies, [0.0], width=0.5)
    assert sharp.max() > broad.max()


def test_converge_starts_at_the_initial_value(Assert):
    result = showcase.converge(3.0, 1.0, number_steps=5)
    assert result.shape == (5,)
    Assert.allclose(result[0], 3.0)


def test_converge_decays_monotonically_onto_the_final_value():
    initial, final = 3.0, 1.0
    result = showcase.converge(initial, final)
    assert result.shape == (showcase.NUMBER_STEPS,)
    assert np.all(np.diff(result) < 0)
    # arriving exactly matters: the final structure of a relaxation has to be the ideal
    # one to the precision the symmetry analysis works with
    assert result[-1] == pytest.approx(final)


def test_decay_falls_from_one_to_zero():
    weights = showcase.decay()
    assert weights.shape == (showcase.NUMBER_STEPS,)
    assert weights[0] == pytest.approx(1.0)
    assert weights[-1] == pytest.approx(0.0)
    assert np.all(np.diff(weights) < 0)


def test_converge_broadcasts_over_arrays(Assert):
    initial = np.zeros((7, 3))
    final = np.arange(21.0).reshape(7, 3)
    result = showcase.converge(initial, final)
    assert result.shape == (showcase.NUMBER_STEPS, 7, 3)
    Assert.allclose(result[0], initial)
    assert np.allclose(result[-1], final, atol=0.01 * np.max(np.abs(final - initial)))
