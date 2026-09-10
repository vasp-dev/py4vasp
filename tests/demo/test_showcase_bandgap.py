# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp._demo import showcase
from py4vasp._demo.showcase import bandgap, band, electronic_structure, kpoint

NUMBER_LABELS = 14


@pytest.fixture
def raw_bandgap():
    return bandgap.Sr2TiO4()


@pytest.fixture
def values(raw_bandgap):
    """Band-edge values of the single spin component, indexed by label."""
    labels = [label.decode() for label in np.array(raw_bandgap.labels)]
    array = np.array(raw_bandgap.values)[:, 0]
    return {label: array[:, index] for index, label in enumerate(labels)}


def test_every_label_names_a_value(raw_bandgap, values):
    shape = np.array(raw_bandgap.values).shape
    assert shape == (showcase.NUMBER_STEPS, 1, NUMBER_LABELS)
    assert len(values) == NUMBER_LABELS


def test_relaxed_gap_is_the_gap_of_the_model(values, Assert):
    fundamental = values["conduction band minimum"] - values["valence band maximum"]
    Assert.allclose(fundamental[-1], electronic_structure.BAND_GAP)


def test_relaxed_band_edges_are_the_extrema_of_the_band_structure(values, Assert):
    # the band structure and the band gap have to tell one story, so the edges are the
    # extrema of the eigenvalues the showcase band structure reports
    model = electronic_structure.Sr2TiO4()
    eigenvalues = model.evaluate(kpoint.mesh())
    valence = eigenvalues[:, : model.number_valence_bands]
    conduction = eigenvalues[:, model.number_valence_bands :]
    Assert.allclose(values["valence band maximum"][-1], np.max(valence))
    Assert.allclose(values["conduction band minimum"][-1], np.min(conduction))


def test_gap_is_indirect(values, raw_bandgap, Assert):
    fundamental = values["conduction band minimum"] - values["valence band maximum"]
    direct = values["direct gap top"] - values["direct gap bottom"]
    assert np.all(fundamental < direct)
    Assert.allclose(_kpoint(values, "VBM")[-1], kpoint.SPECIAL_POINTS["GM"])
    Assert.allclose(_kpoint(values, "CBM")[-1], kpoint.SPECIAL_POINTS["P"])


def test_direct_gap_is_measured_at_a_single_kpoint(values, Assert):
    # both edges of the direct gap belong to the same k point, which is what makes the
    # transition direct; here it is the k point of the conduction band minimum
    Assert.allclose(_kpoint(values, "direct"), _kpoint(values, "CBM"))


def _kpoint(values, name):
    return np.array([values[f"k{axis} ({name})"] for axis in "xyz"]).T


def test_gap_opens_as_the_crystal_relaxes(values):
    fundamental = values["conduction band minimum"] - values["valence band maximum"]
    assert np.all(np.diff(fundamental) > 0)
    # the distortion of the first step closes a noticeable part of the gap
    assert fundamental[0] < 0.9 * fundamental[-1]


def test_band_edges_converge_onto_their_relaxed_values(values):
    for label in ("valence band maximum", "conduction band minimum"):
        deviation = np.abs(values[label] - values[label][-1])
        assert np.all(np.diff(deviation) < 0)
        assert deviation[-1] == 0.0


def test_fermi_energy_is_the_one_of_the_calculation(values, Assert):
    model = electronic_structure.Sr2TiO4()
    Assert.allclose(values["Fermi energy"], model.fermi_energy)


def test_fermi_energy_lies_in_the_gap_at_every_step(values):
    assert np.all(values["valence band maximum"] < values["Fermi energy"])
    assert np.all(values["Fermi energy"] < values["conduction band minimum"])


def test_gap_agrees_with_the_occupations_of_the_band_structure(raw_bandgap, values):
    # the band structure fills exactly the bands below the Fermi energy, so the number
    # of occupied bands must match the valence count the band edges are derived from
    occupations = np.array(band.Sr2TiO4("no_projectors").occupations)
    model = electronic_structure.Sr2TiO4()
    assert np.all(np.sum(occupations[0], axis=-1) == model.number_valence_bands)
