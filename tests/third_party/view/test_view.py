# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import io
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp._third_party.view import View
from py4vasp._third_party.view.view import GridQuantity, IonArrow
from py4vasp._util import import_
from py4vasp.calculation._structure import Structure

ase_cube = import_.optional("ase.io.cube")


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
    if is_structure:
        view = View(**inputs)
        expected_pdb_repr = """\
    CRYST1    4.000    4.000    4.000  90.00  90.00  90.00 P 1
    MODEL     1
    ATOM      1   Sr MOL     1       0.000   0.000   0.000  1.00  0.00          SR
    ATOM      2   Ti MOL     1       2.000   2.000   2.000  1.00  0.00          TI
    ATOM      3    O MOL     1       0.000   2.000   2.000  1.00  0.00           O
    ATOM      4    O MOL     1       2.000   0.000   2.000  1.00  0.00           O
    ATOM      5    O MOL     1       2.000   2.000   0.000  1.00  0.00           O
    ENDMDL
    """
    else:
        view = View(**inputs)
        expected_pdb_repr = """\
CRYST1    3.960    3.960    3.960  60.00  60.00  60.00 P 1
MODEL     1
ATOM      1   Ga MOL     1       0.000   0.000   0.000  1.00  0.00          GA
ATOM      2   As MOL     1       1.980   1.143   0.808  1.00  0.00          AS
ENDMDL
"""
    view.ref = expected_pdb_repr
    return view


@pytest.fixture(params=[True, False])
def view3d(request, not_core):
    is_structure = request.param
    inputs = base_input_view(is_structure)
    if is_structure:
        charge_grid_scalar = GridQuantity(np.random.rand(1, 12, 10, 8), "charge")
        grid_scalars = [charge_grid_scalar]
    else:
        charge_grid_scalar = GridQuantity(np.random.rand(2, 12, 10, 8), "charge")
        potential_grid_scalar = GridQuantity(np.random.rand(2, 12, 10, 8), "potential")
        grid_scalars = [charge_grid_scalar, potential_grid_scalar]
    view = View(grid_scalars=grid_scalars, **inputs)
    view.ref = SimpleNamespace()
    view.ref.grid_scalars = grid_scalars
    return view


@pytest.fixture(params=[True, False])
def view_arrow(request, not_core):
    is_structure = request.param
    inputs = base_input_view(is_structure)
    number_atoms = len(inputs["elements"][0])
    if is_structure:
        force_ion_arrows = IonArrow(np.random.rand(1, number_atoms, 3), "force")
        ion_arrows = [force_ion_arrows]
    else:
        force_ion_arrows = IonArrow(np.random.rand(2, number_atoms, 3), "force")
        moments_ion_arrows = IonArrow(np.random.rand(2, number_atoms, 3), "moments")
        ion_arrows = [force_ion_arrows, moments_ion_arrows]
    view = View(ion_arrows=ion_arrows, **inputs)
    view.ref = SimpleNamespace()
    view.ref.ion_arrows = ion_arrows
    return view


def test_structure_to_view(view, Assert):
    widget = view.to_ngl()
    for idx_traj in range(len(view.lattice_vectors)):
        positions = view.positions[idx_traj]
        lattice_vectors = view.lattice_vectors[idx_traj].T
        expected_coordinates = positions @ lattice_vectors.T
        output_coordinates = widget.trajectory_0.get_coordinates(idx_traj)
        Assert.allclose(expected_coordinates, output_coordinates)
    output_structure_string = widget.trajectory_0.get_structure_string()
    expected_structure_string = view.ref
    expected_lines = expected_structure_string.split("\n")
    output_lines = output_structure_string.split("\n")
    for output_line, expected_line in zip(expected_lines, output_lines):
        assert output_line.strip() == expected_line.strip()


@patch("nglview.NGLWidget._ipython_display_", autospec=True)
def test_ipython(mock_display, view):
    display = view._ipython_display_()
    mock_display.assert_called_once()


def test_isosurface(view3d):
    widget = view3d.to_ngl()
    assert widget.get_state()["_ngl_msg_archive"][1]["args"][0]["binary"] == False
    for idx in range(len(view3d.lattice_vectors)):
        for grid_scalar in view3d.ref.grid_scalars:
            expected_data = grid_scalar.quantity[idx]
            state = widget.get_state()
            output_cube = state["_ngl_msg_archive"][idx + 1]["args"][0]["data"]
            output_data = ase_cube.read_cube(io.StringIO(output_cube))["data"]
            assert expected_data.shape == output_data.shape
            np.allclose(expected_data, output_data)


def test_ion_arrows(view_arrow):
    widget = view_arrow.to_ngl()
    for idx_traj in range(len(view_arrow.lattice_vectors)):
        for idx_ion, ion_arrow in enumerate(view_arrow.ref.ion_arrows):
            ion_positions = view_arrow.positions[idx_traj]
            expected_tail = ion_positions
            expected_tip = ion_arrow.quantity[idx_traj] + ion_positions
            idx_msg = idx_traj + 2 * idx_ion + 1
            msg_archive = widget.get_state()["_ngl_msg_archive"][idx_msg]["args"][1][0]
            output_tail = msg_archive[1]
            output_tip = msg_archive[2][idx_traj]
            assert np.allclose(expected_tip, output_tip)
            assert np.allclose(expected_tail, output_tail)
