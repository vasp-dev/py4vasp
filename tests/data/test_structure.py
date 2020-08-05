from unittest.mock import patch
from py4vasp.data import Structure
import py4vasp.data as data
import py4vasp.raw as raw
import pytest
import numpy as np


@pytest.fixture
def raw_structure():
    number_atoms = 20
    shape = (number_atoms, 3)
    return raw.Structure(
        cell=raw.Cell(scale=1.0, lattice_vectors=np.eye(3)),
        cartesian_positions=np.arange(np.prod(shape)).reshape(shape),
        species=np.array(["C"] * number_atoms, dtype="S2"),
    )


def get_messages_after_structure_information(view):
    message_archive = view.get_state()["_ngl_msg_archive"]
    all_messages = [(msg["methodName"], msg["args"]) for msg in message_archive]
    return all_messages[1:]  # first message is structure data


def test_read(raw_structure, Assert):
    actual = Structure(raw_structure).read()
    assert (actual["cell"] == raw_structure.cell.lattice_vectors).all()
    Assert.allclose(actual["cartesian_positions"], raw_structure.cartesian_positions)
    assert (actual["species"] == raw_structure.species).all()


def test_to_pymatgen(raw_structure):
    structure = Structure(raw_structure)
    mg_structure = structure.to_pymatgen()
    a, b, c = mg_structure.lattice.as_dict()["matrix"]
    assert a == [1, 0, 0]
    assert b == [0, 1, 0]
    assert c == [0, 0, 1]


def test_plot(raw_structure):
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    with cm_init as init, cm_cell as cell:
        structure = Structure(raw_structure)
        structure.plot()
        init.assert_called_once()
        cell.assert_called_once()
