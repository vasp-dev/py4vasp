# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import io
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp._third_party.view import View
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


def test_isosurface(raw_data, Assert, not_core):
    raw_density = raw_data.density("Fe3O4 collinear")
    structure = Structure.from_data(raw_density.structure)
    structure_data = structure.read()
    number_ion_types, ion_types = np.unique(
        structure_data["elements"], return_counts=True
    )
    number_ion_types = np.atleast_2d(number_ion_types)
    ion_types = np.atleast_2d(ion_types)
    view = View(
        number_ion_types=number_ion_types.tolist(),
        ion_types=ion_types.tolist(),
        lattice_vectors=structure_data["lattice_vectors"],
        positions=structure_data["positions"],
        grid_scalars={"charge": raw_density.charge},
    )
    widget = view.show_isosurface("charge")
    assert widget.get_state()["_ngl_msg_archive"][1]["args"][0]["binary"] == False
    output_cube = widget.get_state()["_ngl_msg_archive"][1]["args"][0]["data"]
    output_data = ase_cube.read_cube(io.StringIO(output_cube))["data"]
    expected_data = raw_density.charge[0, ...].astype(np.float32)
    assert expected_data.shape == output_data.shape
    np.allclose(expected_data, output_data)
