# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp._calculation.band import Band
from py4vasp._calculation.kpoint import Kpoint
from py4vasp._demo import showcase
from py4vasp._demo.showcase import band as showcase_band
from py4vasp._demo.showcase import dos as showcase_dos
from py4vasp._demo.showcase import electronic_structure, kpoint


@pytest.fixture
def model():
    return electronic_structure.Sr2TiO4()


@pytest.fixture
def path():
    return kpoint.line_mode()


def test_path_samples_four_contiguous_segments(path, Assert):
    coordinates = np.array(path.coordinates)
    assert coordinates.shape == (4 * showcase.LINE_LENGTH, 3)
    segments = np.split(coordinates, 4)
    for current, following in zip(segments, segments[1:]):
        # the end of one segment is the start of the next, so the path has no jump
        Assert.allclose(current[-1], following[0])


def test_path_is_labelled_at_the_high_symmetry_points(path):
    kpoints = Kpoint.from_data(path)
    assert kpoints.mode() == "line"
    assert kpoints.line_length() == showcase.LINE_LENGTH
    assert kpoints.number_lines() == 4
    labelled = {index: label for index, label in enumerate(kpoints.labels()) if label}
    # every corner is named from both of the segments it belongs to, so a corner shared
    # by two segments appears twice carrying the same name
    gamma = "Γ"
    assert list(labelled.values()) == [gamma, "X", "X", "P", "P", "N", "N", gamma]
    assert list(labelled) == [0, 40, 41, 81, 82, 122, 123, 163]


def test_path_ticks_do_not_mark_a_jump(raw_band):
    # py4vasp joins the labels of two k points at the same distance with a "|" to mark a
    # discontinuity in the path; a contiguous path must not produce one
    pytest.importorskip("plotly")
    gamma = "Γ"
    xticks = Band.from_data(raw_band).to_graph().xticks
    assert list(xticks.values()) == [gamma, "X", "P", "N", gamma]


def test_mesh_covers_the_brillouin_zone_once():
    mesh = kpoint.mesh()
    assert mesh.shape == (np.prod(showcase.KPOINT_GRID), 3)
    assert np.all(mesh >= 0) and np.all(mesh < 1)
    assert len(np.unique(mesh, axis=0)) == len(mesh)


def test_model_opens_a_gap_at_the_fermi_energy(model):
    eigenvalues = model.evaluate(kpoint.mesh())
    valence = eigenvalues[:, : model.number_valence_bands]
    conduction = eigenvalues[:, model.number_valence_bands :]
    assert np.max(valence) == pytest.approx(0.0)
    assert np.min(conduction) == pytest.approx(electronic_structure.BAND_GAP)
    assert model.fermi_energy == pytest.approx(electronic_structure.BAND_GAP / 2)


def test_model_disperses_smoothly_along_the_path(model, path):
    eigenvalues = model.evaluate(np.array(path.coordinates))
    assert eigenvalues.shape == (4 * showcase.LINE_LENGTH, model.number_bands)
    # a band that jumps between neighboring k points draws a broken line
    assert np.max(np.abs(np.diff(eigenvalues, axis=0))) < 0.2


def test_model_never_closes_the_gap(model):
    # the gap is aligned using the analytic extrema of the hopping model, so no k point
    # anywhere in the zone may reach past them and close it
    dense = np.random.default_rng(0).uniform(size=(5000, 3))
    eigenvalues = model.evaluate(dense)
    valence = eigenvalues[:, : model.number_valence_bands]
    conduction = eigenvalues[:, model.number_valence_bands :]
    assert np.max(valence) <= model.valence_band_maximum + 1e-10
    assert np.min(conduction) >= model.conduction_band_minimum - 1e-10


def test_model_has_an_indirect_gap(model, Assert):
    # the valence band maximum sits at Gamma and the conduction band minimum at P, so
    # the band structure along the path actually shows why the gap is indirect
    gamma, p_point = kpoint.SPECIAL_POINTS["GM"], kpoint.SPECIAL_POINTS["P"]
    eigenvalues = model.evaluate(np.array([gamma, p_point]))
    valence, conduction = np.split(eigenvalues, [model.number_valence_bands], axis=1)
    Assert.allclose(np.max(valence[0]), model.valence_band_maximum)
    Assert.allclose(np.min(conduction[1]), model.conduction_band_minimum)


def test_model_respects_the_tetragonal_symmetry(model, Assert):
    # the fourfold axis exchanges the two in-plane translations, so exchanging the
    # first two fractional coordinates must leave the eigenvalues unchanged
    kpoints = np.random.default_rng(1).uniform(size=(100, 3))
    Assert.allclose(model.evaluate(kpoints), model.evaluate(kpoints[:, [1, 0, 2]]))


@pytest.fixture
def raw_band():
    return showcase_band.Sr2TiO4("with_projectors")


def test_band_samples_the_labelled_path(raw_band, model):
    eigenvalues = np.array(raw_band.dispersion.eigenvalues)
    assert eigenvalues.shape == (1, 4 * showcase.LINE_LENGTH, model.number_bands)
    assert not raw_band.dispersion.kpoints.label_indices.is_none()


def test_band_fills_the_valence_and_empties_the_conduction_states(raw_band, model):
    eigenvalues = np.array(raw_band.dispersion.eigenvalues)[0]
    occupations = np.array(raw_band.occupations)[0]
    assert occupations.shape == eigenvalues.shape
    valence, conduction = np.split(occupations, [model.number_valence_bands], axis=1)
    assert np.all(valence == 1)
    assert np.all(conduction == 0)
    # equivalently, every state below the Fermi energy is the occupied one
    assert np.all((eigenvalues < raw_band.fermi_energy) == (occupations == 1))


def test_band_agrees_with_the_density_of_states_on_the_fermi_energy(raw_band):
    # Band and Dos share results/electron_dos/efermi in the file, so a mismatch here
    # would mean whichever is written first silently decides for both
    assert raw_band.fermi_energy == showcase_dos.Sr2TiO4("no_projectors").fermi_energy


def test_band_shows_the_gap_of_the_model(raw_band, model):
    eigenvalues = np.array(raw_band.dispersion.eigenvalues)[0]
    valence, conduction = np.split(eigenvalues, [model.number_valence_bands], axis=1)
    # the path visits both extrema, so it reproduces the gap of the model exactly
    assert np.max(valence) == pytest.approx(model.valence_band_maximum)
    assert np.min(conduction) == pytest.approx(model.conduction_band_minimum)


def test_band_projections_are_normalized_per_state(raw_band, model, Assert):
    projections = np.array(raw_band.projections)
    number_kpoints = 4 * showcase.LINE_LENGTH
    assert projections.shape == (1, 7, 16, number_kpoints, model.number_bands)
    # every state is fully accounted for by the atoms and orbitals it projects onto
    Assert.allclose(
        np.sum(projections, axis=(1, 2)),
        np.ones((1, number_kpoints, model.number_bands)),
    )


def test_band_projections_carry_the_same_character_as_the_dos(raw_band, model):
    projections = np.array(raw_band.projections)[0]
    valence = slice(None, model.number_valence_bands)
    conduction = slice(model.number_valence_bands, None)
    oxygen_p = np.sum(projections[3:7, 1:4, :, valence])
    titanium_t2g = np.sum(projections[2, [4, 5, 7]][:, :, conduction])
    assert oxygen_p / np.sum(projections[:, :, :, valence]) > 0.7
    assert titanium_t2g / np.sum(projections[:, :, :, conduction]) > 0.7


def test_band_without_labels_falls_back_to_coordinates(model):
    raw_band = showcase_band.Sr2TiO4("no_projectors", "no_labels")
    assert raw_band.dispersion.kpoints.label_indices.is_none()
    assert raw_band.projections.is_none()
    # py4vasp then labels the band edges with the coordinates of the k point
    labels = Kpoint.from_data(raw_band.dispersion.kpoints).labels()
    assert labels[0] == "$[0 0 0]$"


def test_raw_band_eigenvalues_are_sorted_like_vasp(raw_band):
    # VASP writes the eigenvalues of every k point in ascending order, so the band
    # structure carries them sorted even though the model keeps them by band
    eigenvalues = np.array(raw_band.dispersion.eigenvalues)[0]
    assert np.all(np.diff(eigenvalues, axis=1) >= 0)


def test_model_keeps_the_eigenvalues_of_a_band_together(model):
    # the density of states attaches an orbital character to every band, so column n has
    # to stay band n; sorting is the job of whoever writes the band structure
    eigenvalues = model.evaluate(kpoint.mesh())
    at_gamma = model.evaluate([kpoint.SPECIAL_POINTS["GM"]])[0]
    assert np.argmax(at_gamma) == model.number_bands - 1
    assert eigenvalues.shape == (len(kpoint.mesh()), model.number_bands)


def test_sort_bands_records_where_every_band_went(Assert):
    # copper is the case that matters: its free-electron band dips below the d bands, so
    # the column a band occupies changes from one k point to the next
    model = electronic_structure.Cu()
    by_band = model.evaluate(kpoint.mesh())
    eigenvalues, order = electronic_structure.sort_bands(by_band)
    assert np.all(np.diff(eigenvalues, axis=1) >= 0)
    assert np.any(order != np.arange(model.number_bands))
    # following the order back recovers the eigenvalue of each band
    Assert.allclose(np.take_along_axis(eigenvalues, np.argsort(order), axis=1), by_band)


@pytest.mark.parametrize(
    "name", sorted({**kpoint.SPECIAL_POINTS, **kpoint.FCC_SPECIAL_POINTS})
)
def test_every_special_point_can_be_labelled(name):
    # a point defined in the module but missing from the label table used to raise a bare
    # KeyError as soon as it was put on a path
    assert kpoint._label(name)
