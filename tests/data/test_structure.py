from unittest.mock import patch
from py4vasp.data import Structure
from .test_topology import raw_topology
import py4vasp.data as data
import py4vasp.raw as raw
import pytest
import numpy as np


@pytest.fixture
def raw_structure(raw_topology):
    number_atoms = len(raw_topology.elements)
    shape = (number_atoms, 3)
    structure = raw.Structure(
        topology=raw_topology,
        cell=raw.Cell(scale=2.0, lattice_vectors=np.eye(3)),
        positions=np.arange(np.prod(shape)).reshape(shape) / np.prod(shape),
    )
    structure.actual_cell = structure.cell.scale * structure.cell.lattice_vectors
    return structure


def test_from_file(raw_structure, mock_file, check_read):
    with mock_file("structure", raw_structure) as mocks:
        check_read(Structure, mocks, raw_structure)


def test_read(raw_structure, Assert):
    actual = Structure(raw_structure).read()
    Assert.allclose(actual["cell"], raw_structure.actual_cell)
    Assert.allclose(actual["positions"], raw_structure.positions)
    assert actual["elements"] == raw_structure.topology.elements


def test_to_ase(raw_structure, Assert):
    structure = Structure(raw_structure).to_ase()
    Assert.allclose(structure.cell.array, raw_structure.actual_cell)
    Assert.allclose(structure.get_scaled_positions(), raw_structure.positions)
    assert all(structure.symbols == "Sr2TiO4")
    assert all(structure.pbc)


def test_tilted_unitcell(raw_structure, Assert):
    cell = np.array([[4, 0, 0], [0, 4, 0], [2, 2, 6]])
    inv_cell = np.linalg.inv(cell)
    cartesian_positions = (
        (0, 0, 0),
        (4, 4, 4),
        (2, 2, 2),
        (2, 2, 0),
        (2, 4, 2),
        (4, 2, 2),
        (2, 2, 4),
    )
    raw_structure.cell = raw.Cell(scale=1, lattice_vectors=cell)
    raw_structure.positions = cartesian_positions @ inv_cell
    structure = Structure(raw_structure).to_ase()
    Assert.allclose(structure.positions, cartesian_positions)


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
    # scale all dimensions by constant factor
    scale = 2
    supercell = structure.to_ase(supercell=scale)
    assert len(supercell) == number_atoms * scale ** 3
    Assert.allclose(supercell.cell.array, raw_structure.actual_cell * scale)
    # scale differently for each dimension
    scale = (2, 1, 3)
    supercell = structure.to_ase(supercell=scale)
    assert len(supercell) == number_atoms * np.prod(scale)
    Assert.allclose(supercell.cell.array, raw_structure.actual_cell * scale)
