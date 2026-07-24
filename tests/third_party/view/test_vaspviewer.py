# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import io
import itertools
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._third_party.view import View
from py4vasp._third_party.view.view import (
    CrystalSymmetry,
    GridQuantity,
    IonArrow,
    Isosurface,
    PhononDispersion,
)
from py4vasp._util import convert, import_

vaspview = import_.optional("vasp.viewer")
hasVaspView = pytest.mark.skipif(
    not import_.is_imported(vaspview),
    reason="vasp_viewer not installed",
)


def _trajectory_from_state(state_dict):
    """Extract a numpy array from a viewer state dict with 'shape' and 'data' keys."""
    return np.frombuffer(bytes(state_dict["data"]), dtype=np.float32).reshape(
        state_dict["shape"]
    )


def _quantity_from_state(state_dict):
    """Extract a numpy array from a viewer state dict with 'quantity_shape' and 'quantity' keys."""
    return np.frombuffer(bytes(state_dict["quantity"]), dtype=np.float32).reshape(
        state_dict["quantity_shape"]
    )


def base_input_view(is_structure):
    if is_structure:
        return {
            "elements": [["Sr", "Ti", "O", "O", "O"]],
            "lattice_vectors": [4 * np.eye(3)],
            "positions": [
                [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.5, 0.5, 0.0],
                ]
            ],
        }
    else:
        return {
            "elements": [["Ga", "As"], ["Ga", "As"]],
            "lattice_vectors": [
                2.8 * (np.ones((3, 3)) - np.eye(3)),
                2.9 * (np.ones((3, 3)) - np.eye(3)),
            ],
            "positions": [
                [
                    [0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [0.26, 0.24, 0.27],
                ],
            ],
        }


@pytest.fixture(params=[True, False])
def view(request):
    is_structure = request.param
    inputs = base_input_view(is_structure)
    view = View(**inputs)
    return view


@hasVaspView
def test_structure_to_view(view: View, Assert):
    state = view.to_vasp_viewer().get_state()
    # check positions, lattice and element types
    assert np.allclose(
        view.positions, _trajectory_from_state(state["_atoms_trajectory"]), atol=1e-7
    )
    Assert.allclose(view.elements, state["_atoms_types"])
    assert np.allclose(
        view.lattice_vectors,
        _trajectory_from_state(state["_lattice_vectors"]),
        atol=1e-7,
    )


@hasVaspView
@patch("vasp.viewer.Widget", autospec=True)
def test_ipython(mock_display, view):
    display = view._ipython_display_(mode="vasp_viewer")
    mock_display.assert_called_once()


@hasVaspView
@patch("vasp.viewer.Widget", autospec=True)
def test_ipython_auto(mock_display, view):
    display = view._ipython_display_(mode="auto")
    mock_display.assert_called_once()


@hasVaspView
@pytest.mark.parametrize("camera", ("orthographic", "perspective"))
def test_camera(view, camera):
    view.camera = camera
    state = view.to_vasp_viewer().get_state()
    assert camera == state["_selections_camera_mode"]


def test_volume_dataset_axis_order():
    # The VASP Viewer uploads the flat buffer into a 3D texture whose first grid
    # axis (grid[0]) varies fastest in memory. Regression test that the config
    # produces data in that layout so the density is not axis-swapped/distorted.
    na, nb, nc = 2, 3, 4
    volume = np.arange(na * nb * nc, dtype=float).reshape(na, nb, nc)
    grid_quantity = GridQuantity(
        quantity=volume[np.newaxis],
        label="charge",
        isosurfaces=[Isosurface(0.5, "#ffffff", 0.6)],
    )
    view = View(
        elements=[["Si"]],
        lattice_vectors=[np.eye(3)],
        positions=[[[0.0, 0.0, 0.0]]],
        grid_scalars=[grid_quantity],
    )
    dataset = view.to_vasp_viewer_config()["volume_datasets"][0]
    grid = tuple(int(value) for value in np.asarray(dataset["grid"]))
    # replicate the viewer's flatten (np.asarray(...).tobytes(), i.e. C order)
    buffer = np.asarray(dataset["data"], dtype=np.float32).ravel()
    nx, ny, nz = grid
    # the viewer reads reconstructed[x, y, z] = buffer[x + nx * y + nx * ny * z]
    reconstructed = buffer.reshape(nz, ny, nx).T
    assert grid == (na, nb, nc)
    assert np.array_equal(reconstructed, volume)


def _phonon_dispersion():
    n_qpoints, n_bands, n_atoms_prim = 4, 6, 2
    size = n_qpoints * n_bands * n_atoms_prim * 3
    real = np.arange(size, dtype=float).reshape(n_qpoints, n_bands, n_atoms_prim, 3)
    eigenvectors = real + 1j * real[::-1]
    return PhononDispersion(
        eigenvectors=eigenvectors,
        frequencies=np.arange(n_qpoints * n_bands, dtype=float).reshape(
            n_qpoints, n_bands
        ),
        qpoints=np.linspace(0.0, 0.5, n_qpoints * 3).reshape(n_qpoints, 3),
        supercell_matrix=np.eye(3),
        primitive_index=np.array([0, 1]),
        path_labels=[[0, "Gamma"], [3, "X"]],
    )


def _phonon_view(phonon):
    return View(
        elements=[["Ga", "As"]],
        lattice_vectors=[4 * np.eye(3)],
        positions=[[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]],
        phonon=phonon,
    )


def test_phonon_dispersion_serialized_to_config():
    phonon = _phonon_dispersion()
    config = _phonon_view(phonon).to_vasp_viewer_config()
    eigenvectors = np.asarray(phonon.eigenvectors)
    Assert = np.testing.assert_allclose
    Assert(np.asarray(config["phonon_eigenvectors_re"]), eigenvectors.real)
    Assert(np.asarray(config["phonon_eigenvectors_im"]), eigenvectors.imag)
    Assert(np.asarray(config["phonon_frequencies"]), np.asarray(phonon.frequencies))
    Assert(np.asarray(config["phonon_qpoints"]), np.asarray(phonon.qpoints))
    Assert(
        np.asarray(config["phonon_supercell_matrix"]),
        np.asarray(phonon.supercell_matrix),
    )
    assert list(config["phonon_primitive_index"]) == [0, 1]
    assert config["phonon_path_labels"] == [[0, "Gamma"], [3, "X"]]


def test_phonon_fields_absent_without_phonon_data():
    config = _phonon_view(None).to_vasp_viewer_config()
    phonon_keys = [key for key in config if key.startswith("phonon_")]
    assert phonon_keys == []


def _single_grid_view(isosurfaces):
    volume = np.arange(2 * 2 * 2, dtype=float).reshape(2, 2, 2)
    return View(
        elements=[["Si"]],
        lattice_vectors=[np.eye(3)],
        positions=[[[0.0, 0.0, 0.0]]],
        grid_scalars=[
            GridQuantity(
                quantity=volume[np.newaxis], label="q", isosurfaces=isosurfaces
            )
        ],
    )


def test_opposite_sign_isolevels_sent_once_with_sign_modes():
    # NICS-style ±v on one field: the data is emitted once, and each isolevel is
    # tagged with a sign mode so the viewer renders the matching lobe.
    view = _single_grid_view(
        [Isosurface(1.0, "#0000ff", 0.6), Isosurface(-1.0, "#ff0000", 0.6)]
    )
    datasets = view.to_vasp_viewer_config()["volume_datasets"]
    assert len(datasets) == 1  # one field, not one dataset per isolevel
    assert datasets[0]["isosurfaces"] == [
        {"iso_value": 1.0, "color_surface": "#0000ff", "sign_mode": "positive"},
        {"iso_value": -1.0, "color_surface": "#ff0000", "sign_mode": "negative"},
    ]


def test_same_sign_isolevels_keep_default_sign_mode():
    view = _single_grid_view(
        [Isosurface(0.2, "#0000ff", 0.6), Isosurface(0.8, "#00ff00", 0.6)]
    )
    isosurfaces = view.to_vasp_viewer_config()["volume_datasets"][0]["isosurfaces"]
    assert [iso["sign_mode"] for iso in isosurfaces] == ["default", "default"]


@hasVaspView
@pytest.mark.skip(reason="Not yet implemented")
def test_isosurface(view):
    assert False


@hasVaspView
@pytest.mark.skip(reason="Not yet implemented")
def test_shifted_isosurface(view):
    assert False


@hasVaspView
@pytest.mark.skip(reason="Not yet implemented")
def test_fail_isosurface(view):
    assert False


@hasVaspView
@pytest.mark.parametrize("is_structure", [True, False])
def test_ion_arrows(is_structure, Assert):
    inputs = base_input_view(is_structure)
    view = View(
        **inputs,
        ion_arrows=[
            IonArrow(
                np.random.rand(
                    len(inputs["positions"]),
                    len(inputs["positions"][0]),
                    3,
                ),
                label="Magnetization",
                color="#00FFFF",
                radius=0.25,
            ),
            IonArrow(
                np.random.rand(
                    len(inputs["positions"]),
                    len(inputs["positions"][0]),
                    3,
                ),
                label="Velocities",
                color="#84FF00",
                radius=0.13,
            ),
        ],
    )
    state = view.to_vasp_viewer().get_state()
    for arrow_group_view, arrow_group_state in zip(
        view.ion_arrows, state["_ion_arrow_groups"]
    ):
        assert np.allclose(
            arrow_group_view.quantity,
            _quantity_from_state(arrow_group_state),
            atol=1e-7,
        )
        assert arrow_group_view.label == arrow_group_state["label"]
        assert arrow_group_view.color == arrow_group_state["base_color"]
        assert arrow_group_view.radius == arrow_group_state["base_radius"]


@hasVaspView
@pytest.mark.parametrize("is_structure", [True, False])
def test_supercell(is_structure, Assert):
    view = View(**base_input_view(is_structure), supercell=(2, 2, 2))
    state = view.to_vasp_viewer().get_state()
    assert np.allclose(
        view.positions, _trajectory_from_state(state["_atoms_trajectory"]), atol=1e-7
    )
    Assert.allclose(view.supercell, state["_selections_bounds"][3:6])


@hasVaspView
@pytest.mark.parametrize(
    ["is_structure", "is_show_cell"],
    [[True, True], [False, True], [True, False], [False, False]],
)
def test_showcell(is_structure, is_show_cell):
    view = View(**base_input_view(is_structure), show_cell=is_show_cell)
    state = view.to_vasp_viewer().get_state()
    assert state["_selections_show_lattice"] == is_show_cell


@hasVaspView
@pytest.mark.parametrize(
    ["is_structure", "is_show_axes"],
    [[True, True], [False, True], [True, False], [False, False]],
)
def test_showaxes(is_structure, is_show_axes):
    view = View(**base_input_view(is_structure), show_axes=is_show_axes)
    state = view.to_vasp_viewer().get_state()
    assert state["_selections_show_abc"] == is_show_axes


@hasVaspView
@pytest.mark.parametrize("is_structure", [True, False])
def test_showaxes_different_origin(is_structure, Assert):
    axes_offset = np.array([0.2, 0.2, 0.2])
    view = View(
        **base_input_view(is_structure), show_axes=True, show_axes_at=axes_offset
    )
    state = view.to_vasp_viewer().get_state()
    Assert.allclose(state["_selections_axes_abc_shift"], axes_offset)
    Assert.allclose(state["_selections_axes_xyz_shift"], axes_offset)


@hasVaspView
@pytest.mark.parametrize("atom_radius", [0.1, 0.5, 1.0])
def test_atom_radius(atom_radius):
    view = View(**base_input_view(is_structure=True), atom_radius=atom_radius)
    state = view.to_vasp_viewer().get_state()
    assert state["_selections_atom_radius"] == atom_radius


@hasVaspView
def test_structure_title():
    view = View(
        **base_input_view(is_structure=True), structure_title="My Structure Title"
    )
    state = view.to_vasp_viewer().get_state()
    assert state["_selections_descriptor"] == view.structure_title


@hasVaspView
def test_different_number_of_steps_raises_error(view):
    too_many_elements = [element for element in view.elements] + [view.elements[0]]
    with pytest.raises(exception.IncorrectUsage):
        View(too_many_elements, view.lattice_vectors, view.positions)
    with pytest.raises(exception.IncorrectUsage):
        broken_view = copy.copy(view)
        broken_view.elements = too_many_elements
        broken_view.to_vasp_viewer()
    #
    too_many_cells = [cell for cell in view.lattice_vectors] + [view.lattice_vectors[0]]
    with pytest.raises(exception.IncorrectUsage):
        View(view.elements, too_many_cells, view.positions)
    with pytest.raises(exception.IncorrectUsage):
        broken_view = copy.copy(view)
        broken_view.lattice_vectors = too_many_cells
        broken_view.to_vasp_viewer()
    #
    too_many_positions = [position for position in view.positions] + [view.positions[0]]
    with pytest.raises(exception.IncorrectUsage):
        View(view.elements, view.lattice_vectors, too_many_positions)
    with pytest.raises(exception.IncorrectUsage):
        broken_view = copy.copy(view)
        broken_view.positions = too_many_positions
        broken_view.to_vasp_viewer()


@hasVaspView
def test_incorrect_shape_raises_error(view):
    different_number_atoms = np.zeros((len(view.positions), 7, 3))
    with pytest.raises(exception.IncorrectUsage):
        View(view.elements, view.lattice_vectors, different_number_atoms)
    not_a_three_component_vector = np.array(view.positions)[:, :, :2]
    with pytest.raises(exception.IncorrectUsage):
        View(view.elements, view.lattice_vectors, not_a_three_component_vector)
    incorrect_unit_cell = np.zeros((len(view.lattice_vectors), 2, 4))
    with pytest.raises(exception.IncorrectUsage):
        View(view.elements, incorrect_unit_cell, view.positions)


def _crystal_symmetry():
    return CrystalSymmetry(
        space_group=225,
        international_symbol="Fm-3m",
        point_group="m-3m",
        crystal_system="cubic",
        is_symmorphic=True,
        equivalent_atoms=np.array([0, 1, 2, 2, 2]),
        wyckoff_letters=["a", "b", "c", "c", "c"],
        wyckoff_site_symmetries=["m-3m", "m-3m", "4/mm.m", "4/mm.m", "4/mm.m"],
    )


def _symmetry_view(crystal_symmetry):
    return View(
        elements=[["Sr", "Ti", "O", "O", "O"]],
        lattice_vectors=[4 * np.eye(3)],
        positions=[
            [[0, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
        ],
        crystal_symmetry=crystal_symmetry,
    )


def test_crystal_symmetry_serialized_to_config():
    config = _symmetry_view(_crystal_symmetry()).to_vasp_viewer_config()
    assert config["crystal_symmetry"] == {
        "space_group": 225,
        "international_symbol": "Fm-3m",
        "point_group": "m-3m",
        "crystal_system": "cubic",
        "is_symmorphic": True,
    }
    assert config["atom_symmetries"] == {
        "equivalent_atoms": [0, 1, 2, 2, 2],
        "wyckoff_letters": ["a", "b", "c", "c", "c"],
        "wyckoff_site_symmetries": ["m-3m", "m-3m", "4/mm.m", "4/mm.m", "4/mm.m"],
    }


def test_symmetry_fields_absent_without_symmetry():
    config = _symmetry_view(None).to_vasp_viewer_config()
    assert "crystal_symmetry" not in config
    assert "atom_symmetries" not in config
