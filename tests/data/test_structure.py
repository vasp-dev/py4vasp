from unittest.mock import patch
from py4vasp.control import POSCAR
from py4vasp.data import Structure, Magnetism
from py4vasp.raw import RawStructure, RawCell, RawVersion
from .test_topology import raw_topology
from .test_magnetism import raw_magnetism, noncollinear_magnetism, charge_only
import py4vasp.data as data
import py4vasp.exceptions as exception
import pytest
import numpy as np

REF_POSCAR = """
Sr2TiO4
1.0
2.0 0.0 0.0
0.0 2.0 0.0
0.0 0.0 2.0
Sr Ti O
2 1 4
Direct
0.023809523809523808 0.07142857142857142 0.11904761904761904
0.16666666666666666 0.21428571428571427 0.2619047619047619
0.30952380952380953 0.35714285714285715 0.40476190476190477
0.4523809523809524 0.5 0.5476190476190477
0.5952380952380952 0.6428571428571429 0.6904761904761905
0.7380952380952381 0.7857142857142857 0.8333333333333334
0.8809523809523809 0.9285714285714286 0.9761904761904762
""".strip()


@pytest.fixture
def raw_structure(raw_topology):
    number_atoms = len(raw_topology.elements)
    shape = (number_atoms, 3)
    structure = RawStructure(
        topology=raw_topology,
        cell=RawCell(scale=2.0, lattice_vectors=np.eye(3)),
        # shift the positions of 0 to avoid relative comparison between tiny numbers
        positions=(0.5 + np.arange(np.prod(shape)).reshape(shape)) / np.prod(shape),
    )
    structure.actual_cell = structure.cell.scale * structure.cell.lattice_vectors
    return structure


def test_from_file(raw_structure, mock_file, check_read):
    with mock_file("structure", raw_structure) as mocks:
        check_read(Structure, mocks, raw_structure)


def test_read(raw_structure, Assert):
    structure = Structure(raw_structure).read()
    check_default_structure(raw_structure, Assert, structure)


def check_default_structure(raw_structure, Assert, structure):
    Assert.allclose(structure["lattice_vectors"], raw_structure.actual_cell)
    Assert.allclose(structure["positions"], raw_structure.positions)
    assert structure["elements"] == raw_structure.topology.elements
    assert "magnetic_moments" not in structure


def test_to_ase(raw_structure, Assert):
    structure = Structure(raw_structure).to_ase()
    Assert.allclose(structure.cell.array, raw_structure.actual_cell)
    Assert.allclose(structure.get_scaled_positions(), raw_structure.positions)
    assert all(structure.symbols == "Sr2TiO4")
    assert all(structure.pbc)


def test_wrong_supercell_size(raw_structure):
    structure = Structure(raw_structure)
    with pytest.raises(exception.IncorrectUsage):
        structure.to_ase("foo")
    with pytest.raises(exception.IncorrectUsage):
        structure.to_ase([1, 2])


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
    raw_structure.cell = RawCell(scale=1, lattice_vectors=cell)
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


def test_magnetism(raw_magnetism, raw_structure, Assert):
    raw_structure.magnetism = raw_magnetism
    structure = Structure(raw_structure)
    expected_moments = raw_magnetism.total_moments[-1]
    Assert.allclose(structure.read()["magnetic_moments"], expected_moments)
    ase = structure.to_ase()
    Assert.allclose(ase.get_initial_magnetic_moments(), expected_moments)
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    cm_arrows = patch.object(data.Viewer3d, "show_arrows_at_atoms")
    with cm_init, cm_cell, cm_arrows as arrows:
        structure.plot()
        arrows.assert_called_once()
        args, kwargs = arrows.call_args
    actual_moments = args[0]
    rescale_moments = Structure.length_moments / np.max(expected_moments)
    for actual, expected in zip(actual_moments, expected_moments):
        Assert.allclose(actual, [0, 0, expected * rescale_moments])


def test_noncollinear_magnetism(noncollinear_magnetism, raw_structure, Assert):
    raw_structure.magnetism = noncollinear_magnetism
    structure = Structure(raw_structure)
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    cm_arrows = patch.object(data.Viewer3d, "show_arrows_at_atoms")
    with cm_init, cm_cell, cm_arrows as arrows:
        structure.plot()
        arrows.assert_called_once()
        args, kwargs = arrows.call_args
    step = -1
    actual_moments = args[0]
    expected_moments = Magnetism(noncollinear_magnetism).total_moments(-1)
    largest_moment = np.max(np.linalg.norm(expected_moments, axis=1))
    rescale_moments = Structure.length_moments / largest_moment
    for actual, expected in zip(actual_moments, expected_moments):
        Assert.allclose(actual, expected * rescale_moments)


def test_charge_only(charge_only, raw_structure, Assert):
    raw_structure.magnetism = charge_only
    structure = Structure(raw_structure)
    assert "magnetic_moments" not in structure.read()
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    cm_arrows = patch.object(data.Viewer3d, "show_arrows_at_atoms")
    with cm_init, cm_cell, cm_arrows as arrows:
        structure.plot()
        arrows.assert_not_called()


def test_vanishing_moments(raw_magnetism, raw_structure, Assert):
    raw_magnetism.moments = np.zeros_like(raw_magnetism.moments)
    raw_structure.magnetism = raw_magnetism
    structure = Structure(raw_structure)
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    cm_arrows = patch.object(data.Viewer3d, "show_arrows_at_atoms")
    with cm_init, cm_cell, cm_arrows as arrows:
        structure.plot()
        arrows.assert_not_called()


def test_print(raw_structure, format_):
    actual, _ = format_(Structure(raw_structure))
    ref_html = """
Sr2TiO4<br>
1.0<br>
<table>
<tr><td>2.0</td><td>0.0</td><td>0.0</td></tr>
<tr><td>0.0</td><td>2.0</td><td>0.0</td></tr>
<tr><td>0.0</td><td>0.0</td><td>2.0</td></tr>
</table>
Sr Ti O<br>
2 1 4<br>
Direct<br>
<table>
<tr><td>0.023809523809523808</td><td>0.07142857142857142</td><td>0.11904761904761904</td></tr>
<tr><td>0.16666666666666666</td><td>0.21428571428571427</td><td>0.2619047619047619</td></tr>
<tr><td>0.30952380952380953</td><td>0.35714285714285715</td><td>0.40476190476190477</td></tr>
<tr><td>0.4523809523809524</td><td>0.5</td><td>0.5476190476190477</td></tr>
<tr><td>0.5952380952380952</td><td>0.6428571428571429</td><td>0.6904761904761905</td></tr>
<tr><td>0.7380952380952381</td><td>0.7857142857142857</td><td>0.8333333333333334</td></tr>
<tr><td>0.8809523809523809</td><td>0.9285714285714286</td><td>0.9761904761904762</td></tr>
</table>""".strip()
    assert actual == {"text/plain": REF_POSCAR, "text/html": ref_html}


def test_from_poscar(raw_structure, Assert):
    structure = Structure.from_POSCAR(REF_POSCAR).read()
    check_default_structure(raw_structure, Assert, structure)


def test_from_ase(raw_structure, Assert):
    structure = Structure(raw_structure).to_ase()
    structure = Structure.from_ase(structure).read()
    check_default_structure(raw_structure, Assert, structure)


def test_descriptor(raw_structure, check_descriptors):
    structure = Structure(raw_structure)
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_to_viewer3d": ["to_viewer3d", "plot"],
        "_to_string": ["to_POSCAR"],
        "_to_ase": ["to_ase"],
    }
    check_descriptors(structure, descriptors)
