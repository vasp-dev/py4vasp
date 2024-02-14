# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import io
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp._third_party.view import View
from py4vasp._third_party.view.view import GridQuantity
from py4vasp._util import import_
from py4vasp.calculation._structure import Structure

ase_cube = import_.optional("ase.io.cube")


@pytest.fixture(params=[True, False])
def view(request, not_core):
    is_structure = request.param
    if is_structure:
        view = View(
            number_ion_types=[[1, 1, 3]],
            ion_types=[["Sr", "Ti", "O"]],
            lattice_vectors=[4 * np.eye(3)],
            positions=[
                [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.5, 0.5, 0.0],
                ]
            ],
        )
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
        view = View(
            number_ion_types=[[1, 1], [1, 1]],
            ion_types=[["Ga", "As"], ["Ga", "As"]],
            lattice_vectors=[
                2.8 * (np.ones((3, 3)) - np.eye(3)),
                2.9 * (np.ones((3, 3)) - np.eye(3)),
            ],
            positions=[
                [
                    [0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [0.26, 0.24, 0.27],
                ],
            ],
        )
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
    if is_structure:
        charge_grid_scalar = GridQuantity(
            quantity=np.random.rand(1, 12, 10, 8), name="charge"
        )
        view = View(
            number_ion_types=[[1, 1, 3]],
            ion_types=[["Sr", "Ti", "O"]],
            lattice_vectors=[4 * np.eye(3)],
            positions=[
                [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.5, 0.5, 0.0],
                ],
            ],
            grid_scalars=[charge_grid_scalar],
        )
    else:
        charge_grid_scalar = GridQuantity(
            quantity=np.random.rand(2, 12, 10, 8), name="charge"
        )
        view = View(
            number_ion_types=[[1, 1], [1, 1]],
            ion_types=[["Ga", "As"], ["Ga", "As"]],
            lattice_vectors=[
                2.8 * (np.ones((3, 3)) - np.eye(3)),
                2.9 * (np.ones((3, 3)) - np.eye(3)),
            ],
            positions=[
                [
                    [0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [0.26, 0.24, 0.27],
                ],
            ],
            grid_scalars=[charge_grid_scalar],
        )
    view.ref = SimpleNamespace()
    view.ref.charge_grid_scalar = charge_grid_scalar
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
    expected_and_output = expected_structure_string.split(
        "\n"
    ), output_structure_string.split("\n")
    for (output_line, expected_line) in zip(*expected_and_output):
        assert output_line.strip() == expected_line.strip()


@patch("nglview.NGLWidget._ipython_display_", autospec=True)
def test_ipython(mock_display, view):
    display = view._ipython_display_()
    mock_display.assert_called_once()


def test_isosurface(view3d, Assert):
    widget = view3d.show_isosurface()
    assert widget.get_state()["_ngl_msg_archive"][1]["args"][0]["binary"] == False
    for idx in range(len(view3d.lattice_vectors)):
        expected_data = view3d.ref.charge_grid_scalar.quantity[idx]
        output_cube = widget.get_state()["_ngl_msg_archive"][idx + 1]["args"][0]["data"]
        output_data = ase_cube.read_cube(io.StringIO(output_cube))["data"]
        assert expected_data.shape == output_data.shape
        np.allclose(expected_data, output_data)
