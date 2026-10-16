# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import io
import itertools
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import _config, exception
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
    displacements = real + 1j * real[::-1]
    return PhononDispersion(
        displacements=displacements,
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
    displacements = np.asarray(phonon.displacements)
    Assert = np.testing.assert_allclose
    Assert(np.asarray(config["phonon_displacements_re"]), displacements.real)
    Assert(np.asarray(config["phonon_displacements_im"]), displacements.imag)
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


def _single_grid_view(isosurfaces, **options):
    volume = np.arange(2 * 2 * 2, dtype=float).reshape(2, 2, 2)
    return View(
        elements=[["Si"]],
        lattice_vectors=[np.eye(3)],
        positions=[[[0.0, 0.0, 0.0]]],
        grid_scalars=[
            GridQuantity(
                quantity=volume[np.newaxis],
                label="q",
                isosurfaces=isosurfaces,
                **options,
            )
        ],
    )


def test_isosurfaces_sent_once_with_the_field():
    # The field is emitted once; each isolevel is a lightweight isosurface entry
    # carrying only its own settings. Sign handling lives on the dataset.
    view = _single_grid_view(
        [Isosurface(1.0, "#0000ff", 0.6), Isosurface(-1.0, "#ff0000", 0.6)]
    )
    datasets = view.to_vasp_viewer_config()["volume_datasets"]
    assert len(datasets) == 1  # one field, not one dataset per isolevel
    common = {"opacity": 0.6, "cap": None, "color_opposite": None}
    assert datasets[0]["isosurfaces"] == [
        {"level": 1.0, "color": "#0000ff", **common},
        {"level": -1.0, "color": "#ff0000", **common},
    ]


@pytest.mark.parametrize("cap", [None, "off"])
def test_isosurface_config_carries_opacity_and_cap(cap):
    # an unset cap is sent as None, so the viewer falls back to its own default
    view = _single_grid_view([Isosurface(0.5, "#0000ff", 0.3, cap=cap)])
    (isosurface,) = view.to_vasp_viewer_config()["volume_datasets"][0]["isosurfaces"]
    assert isosurface["opacity"] == 0.3
    assert isosurface["cap"] == cap


@pytest.mark.parametrize("scheme", [None, "positive", "symmetric"])
def test_volume_config_sends_the_scheme(scheme):
    # the scheme says what the values are, the viewer decides how they look
    options = {} if scheme is None else {"scheme": scheme}
    view = _single_grid_view([Isosurface(1.0, "#0000ff", 0.6)], **options)
    dataset = view.to_vasp_viewer_config()["volume_datasets"][0]
    assert dataset["scheme"] == (scheme or "auto")
    for replaced_key in ("sign_mode", "signed", "symmetric", "color_scheme"):
        assert replaced_key not in dataset  # the viewer refuses them


@pytest.mark.parametrize("colormap", [None, "viridis"])
def test_volume_config_forwards_the_colormap(colormap):
    # an override for hand-built views; the viewer uses it for slices and DVR
    view = _single_grid_view([Isosurface(1.0)], colormap=colormap)
    assert view.to_vasp_viewer_config()["volume_datasets"][0]["colormap"] == colormap


@pytest.mark.parametrize("color_opposite", [None, "#ff8800"])
def test_isosurface_config_forwards_the_opposite_color(color_opposite):
    # the second color of a symmetric or diverging quantity overrides the viewer's
    isosurface = Isosurface(1.0, "#0000ff", 0.6, color_opposite=color_opposite)
    view = _single_grid_view([isosurface], scheme="symmetric")
    dataset = view.to_vasp_viewer_config()["volume_datasets"][0]
    assert dataset["isosurfaces"][0]["color_opposite"] == color_opposite
    assert "color_negative" not in dataset  # the volume carries no colors


def test_volume_config_sends_a_symmetric_isolevel_of_zero():
    # every combination of scheme and isolevel draws something in VASP Viewer
    view = _single_grid_view([Isosurface(0.0, "#0000ff", 0.6)], scheme="symmetric")
    (entry,) = view.to_vasp_viewer_config()["volume_datasets"][0]["isosurfaces"]
    assert entry["level"] == 0.0


def test_negative_symmetric_isolevel_is_sent_as_given():
    # VASP Viewer ignores the sign of a symmetric isolevel, and the colors follow the
    # sign of the surface: color at +0.2, color_opposite at -0.2
    isosurface = Isosurface(-0.2, "#0000ff", 0.6, color_opposite="#ff8800")
    view = _single_grid_view([isosurface], scheme="symmetric")
    (entry,) = view.to_vasp_viewer_config()["volume_datasets"][0]["isosurfaces"]
    assert entry["level"] == -0.2
    assert entry["color"] == "#0000ff"
    assert entry["color_opposite"] == "#ff8800"


@pytest.mark.parametrize("color", [None, "#00ff00"])
def test_isosurface_color_is_sent_only_when_set(color):
    # without a color the viewer picks one from the color scheme
    view = _single_grid_view([Isosurface(0.5, color=color)])
    (entry,) = view.to_vasp_viewer_config()["volume_datasets"][0]["isosurfaces"]
    assert entry["color"] == color


@pytest.mark.parametrize("opacity", [None, 0.3])
def test_isosurface_opacity_is_sent_only_when_set(opacity):
    # without an opacity the viewer draws the isosurface opaque
    options = {} if opacity is None else {"opacity": opacity}
    view = _single_grid_view([Isosurface(0.5, **options)])
    (entry,) = view.to_vasp_viewer_config()["volume_datasets"][0]["isosurfaces"]
    assert entry["opacity"] == opacity


def _view_of_volume(volume):
    grid_quantity = GridQuantity(volume, "q", [Isosurface(0.5, "#0000ff", 0.6)])
    return View(
        elements=[["Si"]],
        lattice_vectors=[np.eye(3)],
        positions=[[[0.0, 0.0, 0.0]]],
        grid_scalars=[grid_quantity],
    )


def test_volume_config_accepts_a_grid_without_the_step_axis():
    # a single grid of shape (Na, Nb, Nc) is the one step VASP Viewer draws
    volume = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    config = _view_of_volume(volume[np.newaxis]).to_vasp_viewer_config()
    with_step_axis = config["volume_datasets"][0]
    config = _view_of_volume(volume).to_vasp_viewer_config()
    without_step_axis = config["volume_datasets"][0]
    assert (
        tuple(without_step_axis["grid"]) == tuple(with_step_axis["grid"]) == (2, 3, 4)
    )
    assert np.allclose(without_step_axis["data"], with_step_axis["data"])


def test_volume_config_checks_a_grid_without_the_step_axis_for_nan():
    volume = np.ones((2, 3, 4))
    volume[1, 2, 3] = np.nan  # outside the first plane
    with pytest.raises(exception.IncorrectUsage):
        _view_of_volume(volume).to_vasp_viewer_config()


@pytest.mark.parametrize(
    "quantity",
    [
        [[[[1.0, None], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]],  # not a number
        [[[[1.0, 2.0], [0.0]], [[0.0, 1.0], [1.0, 0.0]]]],  # ragged
        np.ones((2, 2)),  # not a grid
    ],
)
def test_volume_config_rejects_malformed_grids(quantity):
    with pytest.raises(exception.IncorrectUsage):
        _view_of_volume(quantity).to_vasp_viewer_config()


@pytest.mark.parametrize(
    "quantity",
    [
        np.ones(3),  # no ions
        [[[1, 0, 0], [0, 1]]],  # ragged
        np.ones((1, 5, 2)),  # not three components
    ],
)
def test_vasp_viewer_rejects_malformed_arrows(quantity):
    inputs = base_input_view(is_structure=True)  # 5 ions
    arrow = IonArrow(quantity, "forces", "#FF0000", 0.2)
    view = View(**inputs, ion_arrows=[arrow])
    with pytest.raises(exception.IncorrectUsage):
        view.to_vasp_viewer_config()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_volume_config_rejects_non_finite_values(value):
    # VASP Viewer only draws finite values
    volume = np.ones((1, 2, 2, 2))
    volume[0, 1, 0, 1] = value
    with pytest.raises(exception.IncorrectUsage):
        _view_of_volume(volume).to_vasp_viewer_config()


def test_volume_without_isosurfaces_sends_none():
    # the viewer rejects an empty list, and every volume is sent with the same keys
    dataset = _single_grid_view(None).to_vasp_viewer_config()["volume_datasets"][0]
    assert dataset["isosurfaces"] is None


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
    view = View(**inputs, ion_arrows=_ion_arrows(inputs))
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


def _ion_arrows(inputs):
    shape = (len(inputs["positions"]), len(inputs["positions"][0]), 3)
    return [
        IonArrow(
            np.random.rand(*shape), label="Magnetization", color="#00FFFF", radius=0.25
        ),
        IonArrow(
            np.random.rand(*shape), label="Velocities", color="#84FF00", radius=0.13
        ),
    ]


@pytest.mark.parametrize("is_structure", [True, False])
def test_arrow_group_config_has_no_radius(is_structure):
    # the radius sizes an arrow in NGL, VASP Viewer sizes it from its width mode instead
    inputs = base_input_view(is_structure)
    view = View(**inputs, ion_arrows=_ion_arrows(inputs))
    for group in view.to_vasp_viewer_config()["ion_arrow_groups"]:
        assert "base_radius" not in group


@pytest.mark.parametrize(
    "modes", [{}, {"length_mode": "physical", "width_mode": "atom"}]
)
def test_arrow_group_config_forwards_the_modes(modes):
    # an unset mode is sent as None, so the viewer falls back to its own default
    inputs = base_input_view(is_structure=True)
    arrow = IonArrow(
        np.zeros((1, 5, 3)), label="forces", color="purple", radius=0.2, **modes
    )
    view = View(**inputs, ion_arrows=[arrow])
    (group,) = view.to_vasp_viewer_config()["ion_arrow_groups"]
    assert group["length_mode"] == modes.get("length_mode")
    assert group["width_mode"] == modes.get("width_mode")


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


def test_vasp_viewer_needs_one_arrow_per_ion():
    # only VASP Viewer refuses the mismatch; NGL keeps cycling the arrows over the ions
    inputs = base_input_view(is_structure=True)  # 5 ions
    arrow = IonArrow(np.zeros((1, 4, 3)), "forces", "#FF0000", 0.2)
    view = View(**inputs, ion_arrows=[arrow])
    with pytest.raises(exception.IncorrectUsage):
        view.to_vasp_viewer_config()


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
