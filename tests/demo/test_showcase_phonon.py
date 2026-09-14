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


def test_frequencies_are_sorted_like_vasp(frequencies):
    # VASP writes the frequencies of every q point in ascending order; the acoustic
    # branches overtake the lower optical ones away from Gamma, so concatenating the two
    # groups is not enough
    assert np.all(np.diff(frequencies, axis=1) >= 0)


def test_eigenvectors_follow_their_frequency(raw_band, frequencies, Assert):
    # the pattern in row n has to belong to the frequency in column n, or animating the
    # lowest mode past a crossing shows the displacement of a different one
    eigenvectors = np.array(raw_band.eigenvectors)
    at_gamma, at_boundary = eigenvectors[0], eigenvectors[showcase.LINE_LENGTH - 1]
    # the three acoustic branches are lowest at Gamma but not at the zone boundary, so
    # the rows in those slots differ between the two q points
    assert not np.allclose(at_gamma[:3], at_boundary[:3])


@pytest.fixture
def raw_dos():
    return phonon.dos_Sr2TiO4()


@pytest.fixture
def dos(raw_dos):
    return np.array(raw_dos.energies), np.array(raw_dos.dos)


def test_dos_spans_the_whole_dispersion(dos, frequencies):
    energies, _ = dos
    assert energies[0] == 0.0  # a stable crystal has no state below zero frequency
    assert energies[-1] > np.max(frequencies)
    assert len(energies) == showcase.NUMBER_POINTS


def test_dos_is_positive_everywhere(dos):
    _, values = dos
    assert np.all(values >= 0.0)


def test_dos_integrates_to_the_number_of_modes(dos):
    energies, values = dos
    # the energy axis starts at zero, so the part of the broadening that the modes at
    # the zone centre spread below it is missing; everything else is accounted for
    integral = np.trapezoid(values, energies)
    assert abs(integral / NUMBER_MODES - 1) < 1e-3


def test_dos_ends_where_the_highest_branch_does(dos, frequencies):
    energies, values = dos
    occupied = energies[values > 0.01 * np.max(values)]
    highest = np.max(frequencies)
    # the edge is the highest frequency of the dispersion, smeared by the broadening
    assert highest < occupied[-1] < highest + 4 * phonon.BROADENING
    assert np.all(values[energies > highest + 4 * phonon.BROADENING] < 1e-3)


def test_projections_add_up_to_the_total(raw_dos, Assert):
    projections = np.array(raw_dos.projections)
    assert projections.shape == (phonon.NUMBER_ATOMS, 3, showcase.NUMBER_POINTS)
    Assert.allclose(np.sum(projections, axis=(0, 1)), np.array(raw_dos.dos))


def _share_of_element(raw_dos, element, window):
    projections = np.array(raw_dos.projections)[:, :, window]
    elements = np.array(["Sr", "Sr", "Ti", "O", "O", "O", "O"])
    selected = np.sum(projections[elements == element])
    return selected / np.sum(projections)


def test_acoustic_region_is_carried_by_the_heavy_atom(raw_dos):
    # the acoustic modes translate the whole cell, so the mass-weighted eigenvector
    # puts most of the weight on strontium, the heaviest atom of the crystal
    energies = np.array(raw_dos.energies)
    window = energies < np.min(phonon.OPTICAL_AT_GAMMA) - 4 * phonon.BROADENING
    assert _share_of_element(raw_dos, "Sr", window) > 0.5


def test_top_of_the_spectrum_is_carried_by_oxygen(raw_dos):
    # the highest branches are the oxygen stretching modes of the oxide
    energies = np.array(raw_dos.energies)
    window = energies > max(phonon.OPTICAL_AT_GAMMA) - phonon.BROADENING
    assert _share_of_element(raw_dos, "O", window) > 0.9


def test_stoichiometry_names_the_projected_atoms(raw_dos):
    stoichiometry = raw_dos.stoichiometry
    assert np.sum(np.array(stoichiometry.number_ion_types)) == phonon.NUMBER_ATOMS


def test_eigenvector_weight_follows_the_atomic_masses(Assert):
    # VASP reports mass-weighted displacements, so an acoustic mode that moves every
    # atom by the same amount carries a weight proportional to the mass
    weight = phonon.mode_weights()[0]
    Assert.allclose(weight / weight[0], np.array(phonon.MASSES) / phonon.MASSES[0])


def test_mode_weights_are_normalized(Assert):
    Assert.allclose(np.sum(phonon.mode_weights(), axis=1), np.ones(NUMBER_MODES))


def test_optical_weight_moves_from_the_cation_to_the_anion(Assert):
    oxygen = np.sum(phonon.mode_weights()[3:, 3:], axis=1)
    assert np.all(np.diff(oxygen) > 0)
    assert oxygen[0] < 0.05 and oxygen[-1] > 0.9


@pytest.fixture
def raw_mode():
    return phonon.mode_Sr2TiO4()


@pytest.fixture
def mode_frequencies(raw_mode):
    """The complex frequencies VASP stores as pairs of reals, in THz."""
    complex_ev = np.array(raw_mode.frequencies).flatten().view(np.complex128)
    return complex_ev * phonon.EV_TO_THZ


def test_mode_reports_one_frequency_per_degree_of_freedom(mode_frequencies):
    assert mode_frequencies.shape == (NUMBER_MODES,)


def test_no_mode_is_unstable(mode_frequencies):
    # VASP reports an unstable mode as an imaginary frequency, so a crystal at its
    # energy minimum has none
    assert np.all(mode_frequencies.imag == 0.0)
    assert np.all(mode_frequencies.real >= 0.0)


def test_three_modes_translate_the_crystal(mode_frequencies, Assert):
    Assert.allclose(np.sort(mode_frequencies.real)[:3], np.zeros(3))
    assert np.all(np.sort(mode_frequencies.real)[3:] > 0.0)


def test_mode_frequencies_match_the_dispersion_at_gamma(mode_frequencies, raw_band):
    # the path starts at Gamma, so the first q point of the dispersion is the same one
    frequencies = np.array(raw_band.dispersion.eigenvalues)
    np.testing.assert_allclose(
        np.sort(mode_frequencies.real), np.sort(frequencies[0]), atol=1e-10
    )


def test_mode_displacements_are_normalized(raw_mode, Assert):
    eigenvectors = np.array(raw_mode.eigenvectors)
    assert eigenvectors.shape == (NUMBER_MODES, NUMBER_MODES)
    Assert.allclose(np.sum(eigenvectors**2, axis=1), np.ones(NUMBER_MODES))


def test_mode_displacements_carry_the_share_of_their_mode(raw_mode, Assert):
    # the displacement pattern of a mode has to distribute the mode over the atoms the
    # same way the projected density of states does
    eigenvectors = np.array(raw_mode.eigenvectors)
    per_atom = np.sum(
        eigenvectors.reshape(NUMBER_MODES, phonon.NUMBER_ATOMS, 3) ** 2, axis=2
    )
    Assert.allclose(per_atom, phonon.mode_weights())


def test_mode_describes_the_relaxed_structure(raw_mode, Assert):
    from py4vasp._demo.showcase import structure

    positions = np.array(raw_mode.structure.positions)
    Assert.allclose(positions[-1], structure.ideal_positions())


def test_spectrum_is_broadened_once(raw_dos):
    # every example in the documentation builds the demo data again in the same process
    energies, projections = phonon._spectrum()
    assert not projections.flags.writeable
    assert phonon._spectrum()[1] is projections
    assert np.array(raw_dos.projections) is not projections
