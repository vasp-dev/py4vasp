# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp._demo import showcase
from py4vasp._demo.showcase import phonon

NUMBER_MODES = 21  # three per atom of Sr2TiO4


@pytest.fixture
def raw_band():
    return phonon.band_Sr2TiO4()


@pytest.fixture
def frequencies(raw_band):
    return np.array(raw_band.dispersion.eigenvalues)


def test_phonon_band_samples_the_labelled_path(raw_band, frequencies):
    assert frequencies.shape == (4 * showcase.LINE_LENGTH, NUMBER_MODES)
    assert not raw_band.dispersion.kpoints.label_indices.is_none()


def test_no_frequency_is_imaginary(frequencies):
    # VASP reports an unstable mode as a negative frequency; a relaxed structure has none
    assert np.all(frequencies >= 0)


def test_three_acoustic_branches_vanish_at_gamma(frequencies, Assert):
    at_gamma = frequencies[0]
    # the three translations of the crystal cost no energy at the zone centre
    Assert.allclose(at_gamma[:3], np.zeros(3))
    assert np.all(at_gamma[3:] > 1.0)


def test_acoustic_branches_are_linear_near_gamma(frequencies):
    # sound waves have a linear dispersion, so the slope is finite rather than flat
    acoustic = frequencies[:5, :3]
    slope = np.diff(acoustic, axis=0)
    assert np.all(slope > 0)
    # and the curvature is small compared to the slope over the same range
    assert np.max(np.abs(np.diff(slope, axis=0))) < np.max(slope)


def test_optical_branches_stay_above_the_acoustic_ones(frequencies):
    assert np.all(np.max(frequencies[:, :3]) < np.max(frequencies[:, 3:]))


def test_eigenvectors_are_normalized(raw_band, Assert):
    eigenvectors = np.array(raw_band.eigenvectors)
    assert eigenvectors.shape == (4 * showcase.LINE_LENGTH, NUMBER_MODES, 7, 3, 2)
    real, imaginary = np.moveaxis(eigenvectors, -1, 0)
    norm = np.sum(real**2 + imaginary**2, axis=(-1, -2))
    Assert.allclose(norm, np.ones_like(norm))
