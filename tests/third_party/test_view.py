# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from unittest.mock import patch

import numpy as np
import pytest

from py4vasp._third_party.view import View
from py4vasp.calculation._structure import Structure


@pytest.fixture
def view(not_core):
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
    return view


def test_structure_to_view(view):
    expected_pdb_repr = """\
CRYST1    4.000    4.000    4.000  90.00  90.00  90.00 P 1
MODEL     1
ATOM      1   Sr MOL     1       0.000   0.000   0.000  1.00  0.00          SR
ATOM      2   Ti MOL     1       0.500   0.500   0.500  1.00  0.00          TI
ATOM      3    O MOL     1       0.000   0.500   0.500  1.00  0.00           O
ATOM      4    O MOL     1       0.500   0.000   0.500  1.00  0.00           O
ATOM      5    O MOL     1       0.500   0.500   0.000  1.00  0.00           O
ENDMDL
"""
    widget = view.to_ngl()
    state = widget.get_state()
    assert len(state["_ngl_msg_archive"]) == 1
    output_pdb_repr = state["_ngl_msg_archive"][0]["args"][0]["data"]
    expected_and_output = output_pdb_repr.split("\n"), expected_pdb_repr.split("\n")
    for (output_line, expected_line) in zip(*expected_and_output):
        assert output_line.strip() == expected_line.strip()
    assert state["_ngl_msg_archive"][0]["kwargs"]["ext"] == "pdb"


@patch("nglview.NGLWidget._ipython_display_", autospec=True)
def test_ipython(mock_display, view):
    display = view._ipython_display_()
    mock_display.assert_called_once()
