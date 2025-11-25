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

vaspview = import_.optional("vasp_viewer")
hasVaspView = pytest.mark.skipif(
    not import_.is_imported(vaspview),
    reason="vasp_viewer not installed",
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
def view(request, not_core):
    is_structure = request.param
    inputs = base_input_view(is_structure)
    view = View(**inputs)
    return view


@hasVaspView
def test_structure_to_view(view: View, Assert, not_core):
    state = view.to_vasp_viewer().get_state()
    # check positions, lattice and element types
    Assert.allclose(view.positions, state["atoms_trajectory"])
    Assert.allclose(view.elements[0], state["atoms_types"])
    Assert.allclose(view.lattice_vectors[0], state["lattice_vectors"])


@hasVaspView
@patch("vasp_viewer.Widget", autospec=True)
def test_ipython(mock_display, view, not_core):
    display = view._ipython_display_(mode="vasp_viewer")
    mock_display.assert_called_once()


@hasVaspView
@patch("vasp_viewer.Widget", autospec=True)
def test_ipython_auto(mock_display, view, not_core):
    display = view._ipython_display_(mode="auto")
    mock_display.assert_called_once()


@hasVaspView
@pytest.mark.parametrize("camera", ("orthographic", "perspective"))
def test_camera(view, camera, not_core):
    view.camera = camera
    state = view.to_vasp_viewer().get_state()
    assert camera == state["selections_camera_mode"]


@hasVaspView
@pytest.mark.skip(reason="Not yet implemented")
def test_isosurface(view, not_core):
    assert False


@hasVaspView
@pytest.mark.skip(reason="Not yet implemented")
def test_shifted_isosurface(view, not_core):
    assert False


@hasVaspView
@pytest.mark.skip(reason="Not yet implemented")
def test_fail_isosurface(view, not_core):
    assert False


@hasVaspView
@pytest.mark.skip(reason="Not yet implemented")
def test_ion_arrows(view, not_core):
    assert False


@hasVaspView
@pytest.mark.skip(reason="Not yet implemented")
def test_shifted_ion_arrows(view, not_core):
    assert False


@hasVaspView
@pytest.mark.parametrize("is_structure", [True, False])
def test_supercell(is_structure, Assert, not_core):
    view = View(**base_input_view(is_structure), supercell=(2, 2, 2))
    state = view.to_vasp_viewer().get_state()
    Assert.allclose(view.positions, state["atoms_trajectory"])
    Assert.allclose(view.supercell, state["selections_supercell"])


@hasVaspView
@pytest.mark.parametrize(
    ["is_structure", "is_show_cell"],
    [[True, True], [False, True], [True, False], [False, False]],
)
def test_showcell(is_structure, is_show_cell, not_core):
    view = View(**base_input_view(is_structure), show_cell=is_show_cell)
    state = view.to_vasp_viewer().get_state()
    assert state["selections_show_lattice"] == is_show_cell


@hasVaspView
@pytest.mark.parametrize(
    ["is_structure", "is_show_axes"],
    [[True, True], [False, True], [True, False], [False, False]],
)
def test_showaxes(is_structure, is_show_axes, not_core):
    view = View(**base_input_view(is_structure), show_axes=is_show_axes)
    state = view.to_vasp_viewer().get_state()
    assert state["selections_show_xyz"] == False
    assert state["selections_show_abc"] == is_show_axes
    assert state["selections_show_xyz_aside"] == is_show_axes
    assert state["selections_show_abc_aside"] == is_show_axes


@hasVaspView
@pytest.mark.parametrize("is_structure", [True, False])
def test_showaxes_different_origin(is_structure, Assert, not_core):
    axes_offset = np.array([0.2, 0.2, 0.2])
    view = View(
        **base_input_view(is_structure), show_axes=True, show_axes_at=axes_offset
    )
    state = view.to_vasp_viewer().get_state()
    Assert.allclose(state["selections_axes_abc_shift"], axes_offset)
    Assert.allclose(state["selections_axes_xyz_shift"], axes_offset)


@hasVaspView
@pytest.mark.parametrize("atom_radius", [0.1, 0.5, 1.0])
def test_atom_radius(atom_radius, not_core):
    view = View(**base_input_view(is_structure=True), atom_radius=atom_radius)
    state = view.to_vasp_viewer().get_state()
    assert state["selections_atom_radius"] == atom_radius


@hasVaspView
def test_different_number_of_steps_raises_error(view, not_core):
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
def test_incorrect_shape_raises_error(view, not_core):
    different_number_atoms = np.zeros((len(view.positions), 7, 3))
    with pytest.raises(exception.IncorrectUsage):
        View(view.elements, view.lattice_vectors, different_number_atoms)
    not_a_three_component_vector = np.array(view.positions)[:, :, :2]
    with pytest.raises(exception.IncorrectUsage):
        View(view.elements, view.lattice_vectors, not_a_three_component_vector)
    incorrect_unit_cell = np.zeros((len(view.lattice_vectors), 2, 4))
    with pytest.raises(exception.IncorrectUsage):
        View(view.elements, incorrect_unit_cell, view.positions)
