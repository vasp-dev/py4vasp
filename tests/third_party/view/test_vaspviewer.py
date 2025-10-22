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
from py4vasp._third_party.view.view import GridQuantity, IonArrow, Isosurface
from py4vasp._util import convert, import_


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
def view(request, not_core):
    is_structure = request.param
    inputs = base_input_view(is_structure)
    view = View(**inputs)
    return view


def test_structure_to_view(view: View, Assert):
    widget = view.to_vasp_viewer()
    # check positions
    for idx_traj in range(len(view.lattice_vectors)):
        positions = view.positions[idx_traj]
        expected_coordinates = positions
        output_coordinates = widget.get_state()["atoms_trajectory"][idx_traj]
        Assert.allclose(expected_coordinates, output_coordinates)
    # check lattice
    output_lattice = widget.get_state()["lattice_vectors"]
    expected_lattice = view.lattice_vectors[0]
    Assert.allclose(expected_lattice, output_lattice)
    # check elements
    output_elements = widget.get_state()["atoms_types"]
    expected_elements = view.elements[0]
    Assert.allclose(expected_elements, output_elements)


@patch("vasp_viewer.Widget", autospec=True)
def test_ipython(mock_display, view):
    display = view._ipython_display_(mode="vasp_viewer")
    mock_display.assert_called_once()


@patch("vasp_viewer.Widget", autospec=True)
def test_ipython_auto(mock_display, view):
    display = view._ipython_display_(mode="auto")
    mock_display.assert_called_once()


@pytest.mark.parametrize("camera", ("orthographic", "perspective"))
def test_camera(view, camera):
    view.camera = camera
    widget = view.to_vasp_viewer()
    camera_message = widget.get_state()["selections_camera_mode"]
    assert camera_message == camera


@pytest.mark.skip(reason="Not yet implemented")
def test_isosurface(view):
    assert False


@pytest.mark.skip(reason="Not yet implemented")
def test_shifted_isosurface(view):
    assert False


@pytest.mark.skip(reason="Not yet implemented")
def test_fail_isosurface(view):
    assert False


@pytest.mark.skip(reason="Not yet implemented")
def test_ion_arrows(view):
    assert False


@pytest.mark.skip(reason="Not yet implemented")
def test_shifted_ion_arrows(view):
    assert False


@pytest.mark.parametrize("is_structure", [True, False])
def test_supercell(is_structure, not_core):
    inputs = base_input_view(is_structure)
    supercell = (2, 2, 2)
    inputs["supercell"] = supercell
    view = View(**inputs)
    widget = view.to_vasp_viewer()
    for idx_traj in range(len(inputs["lattice_vectors"])):
        positions = view.positions[idx_traj]
        expected_coordinates = positions
        output_coordinates = widget.get_state()["atoms_trajectory"][idx_traj]
        assert np.allclose(expected_coordinates, output_coordinates)

    assert np.allclose(view.supercell, widget.get_state()["selections_supercell"])


@pytest.mark.parametrize(
    ["is_structure", "is_show_cell"],
    [[True, True], [False, True], [True, False], [False, False]],
)
def test_showcell(is_structure, is_show_cell, not_core):
    inputs = base_input_view(is_structure)
    inputs["show_cell"] = is_show_cell
    view = View(**inputs)
    widget = view.to_vasp_viewer()
    assert widget.get_state()["selections_show_lattice"] == is_show_cell


@pytest.mark.parametrize(
    ["is_structure", "is_show_axes"],
    [[True, True], [False, True], [True, False], [False, False]],
)
def test_showaxes(is_structure, is_show_axes, not_core):
    inputs = base_input_view(is_structure)
    inputs["show_axes"] = is_show_axes
    view = View(**inputs)
    widget = view.to_vasp_viewer()
    assert widget.get_state()["selections_show_xyz"] == False
    assert widget.get_state()["selections_show_abc"] == is_show_axes
    assert widget.get_state()["selections_show_xyz_aside"] == is_show_axes
    assert widget.get_state()["selections_show_abc_aside"] == is_show_axes


@pytest.mark.parametrize("is_structure", [True, False])
def test_showaxes_different_origin(is_structure, not_core):
    inputs = base_input_view(is_structure)
    inputs["show_axes"] = True
    inputs["show_axes_at"] = np.array([0.2, 0.2, 0.2])
    view = View(**inputs)
    widget = view.to_vasp_viewer()
    assert (
        widget.get_state()["selections_axes_abc_shift"]
        == inputs["show_axes_at"].tolist()
    )
    assert (
        widget.get_state()["selections_axes_xyz_shift"]
        == inputs["show_axes_at"].tolist()
    )


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
