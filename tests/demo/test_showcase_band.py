# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp._calculation.kpoint import Kpoint
from py4vasp._demo import showcase
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
    labels = kpoints.labels()
    labelled = {index: label for index, label in enumerate(labels) if label}
    assert list(labelled.values()) == [r"$\Gamma$", "X", "P", "N", r"$\Gamma$"]
    assert list(labelled) == [
        0,
        *(showcase.LINE_LENGTH * step - 1 for step in (1, 2, 3, 4)),
    ]


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


def test_model_orders_eigenvalues_like_vasp(model):
    # VASP writes the eigenvalues of every k point in ascending order
    eigenvalues = model.evaluate(kpoint.mesh())
    assert np.all(np.diff(eigenvalues, axis=1) >= 0)


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
