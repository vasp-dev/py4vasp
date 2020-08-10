from unittest.mock import patch
from py4vasp.data import Structure
import py4vasp.data as data
import py4vasp.raw as raw
import pytest
import numpy as np


@pytest.fixture
def raw_structure():
    number_atoms = 6
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
    Assert.allclose(actual["cell"], raw_structure.cell.lattice_vectors)
    Assert.allclose(actual["cartesian_positions"], raw_structure.cartesian_positions)
    assert (actual["species"] == raw_structure.species).all()


def test_to_ase(raw_structure, Assert):
    structure = Structure(raw_structure).to_ase()
    Assert.allclose(structure.cell.array, raw_structure.cell.lattice_vectors)
    Assert.allclose(structure.positions, raw_structure.cartesian_positions)
    assert all(structure.symbols == "C6")
    assert all(structure.pbc)


def test_plot(raw_structure):
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    with cm_init as init, cm_cell as cell:
        structure = Structure(raw_structure)
        structure.plot()
        init.assert_called_once()
        cell.assert_called_once()


def test_supercell(raw_structure, Assert):
    structure = Structure(raw_structure)
    number_atoms = len(structure)
    cell = raw_structure.cell.lattice_vectors
    # scale all dimensions by constant factor
    scale = 2
    supercell = structure.to_ase(supercell=scale)
    assert len(supercell) == number_atoms * scale ** 3
    Assert.allclose(supercell.cell.array, cell * scale)
    # scale differently for each dimension
    scale = (2, 1, 3)
    supercell = structure.to_ase(supercell=scale)
    assert len(supercell) == number_atoms * np.prod(scale)
    Assert.allclose(supercell.cell.array, cell * scale)
