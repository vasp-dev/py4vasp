from py4vasp.data import Trajectory, Topology
from unittest.mock import patch
import py4vasp.exceptions as exception
import py4vasp.data as data
import pytest
import types
import numpy as np


@pytest.fixture
def Sr2TiO4(raw_data):
    raw_trajectory = raw_data.trajectory("Sr2TiO4")
    trajectory = Trajectory(raw_trajectory)
    trajectory.ref = types.SimpleNamespace()
    trajectory.ref.names = Topology(raw_trajectory.topology).names()
    trajectory.ref.elements = Topology(raw_trajectory.topology).elements()
    trajectory.ref.positions = raw_trajectory.positions
    trajectory.ref.lattice_vectors = raw_trajectory.lattice_vectors
    return trajectory


def test_read_trajectory(Sr2TiO4, Assert):
    trajectory = Sr2TiO4.read()
    assert trajectory["names"] == Sr2TiO4.ref.names
    assert trajectory["elements"] == Sr2TiO4.ref.elements
    Assert.allclose(trajectory["positions"], Sr2TiO4.ref.positions)
    Assert.allclose(trajectory["lattice_vectors"], Sr2TiO4.ref.lattice_vectors)


def test_to_mdtraj(Sr2TiO4, Assert):
    trajectory = Sr2TiO4.to_mdtraj()
    assert trajectory.n_frames == len(Sr2TiO4.ref.positions)
    assert trajectory.n_atoms == len(Sr2TiO4.ref.elements)
    unitcell_vectors = Trajectory.A_to_pm * Sr2TiO4.ref.lattice_vectors
    cartesian_positions = [
        pos @ cell for pos, cell in zip(Sr2TiO4.ref.positions, unitcell_vectors)
    ]
    Assert.allclose(trajectory.xyz, np.array(cartesian_positions).astype(np.float32))
    Assert.allclose(trajectory.unitcell_vectors, unitcell_vectors)


def test_plot(Sr2TiO4):
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    with cm_init as init, cm_cell as cell:
        Sr2TiO4.plot()
        init.assert_called_once()
        cell.assert_called_once()


def test_to_structure(Sr2TiO4, Assert):
    structure = Sr2TiO4.to_structure(0).read()
    assert structure["elements"] == Sr2TiO4.ref.elements
    Assert.allclose(structure["lattice_vectors"], Sr2TiO4.ref.lattice_vectors[0])
    Assert.allclose(structure["positions"], Sr2TiO4.ref.positions[0])


def test_incorrect_step(Sr2TiO4):
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.to_structure(100)
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.to_structure([0, 1])


def test_print(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4)
    ref_plain = """
current structure of 4 step trajectory
1.0
6.9229 0.0 0.0
4.694503016799998 5.0880434191000035 0.0
-5.808696220500002 -2.544019393599997 2.7773292841999986
Sr Ti O
2 1 4
Direct
0.64529 0.64529 0.0
0.35471 0.35471 0.0
0.0 0.0 0.0
0.84178 0.84178 0.0
0.15823 0.15823 0.0
0.5 0.0 0.5
0.0 0.5 0.5
    """.strip()
    ref_html = """
current structure of 4 step trajectory<br>
1.0<br>
<table>
<tr><td>6.9229</td><td>0.0</td><td>0.0</td></tr>
<tr><td>4.694503016799998</td><td>5.0880434191000035</td><td>0.0</td></tr>
<tr><td>-5.808696220500002</td><td>-2.544019393599997</td><td>2.7773292841999986</td></tr>
</table>
Sr Ti O<br>
2 1 4<br>
Direct<br>
<table>
<tr><td>0.64529</td><td>0.64529</td><td>0.0</td></tr>
<tr><td>0.35471</td><td>0.35471</td><td>0.0</td></tr>
<tr><td>0.0</td><td>0.0</td><td>0.0</td></tr>
<tr><td>0.84178</td><td>0.84178</td><td>0.0</td></tr>
<tr><td>0.15823</td><td>0.15823</td><td>0.0</td></tr>
<tr><td>0.5</td><td>0.0</td><td>0.5</td></tr>
<tr><td>0.0</td><td>0.5</td><td>0.5</td></tr>
</table>
    """.strip()
    assert actual == {"text/plain": ref_plain, "text/html": ref_html}


def test_descriptor(Sr2TiO4, check_descriptors):
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_to_viewer3d": ["to_viewer3d", "plot"],
        "_to_structure": ["to_structure"],
        "_to_mdtraj": ["to_mdtraj"],
    }
    check_descriptors(Sr2TiO4, descriptors)


def test_from_file(raw_data, mock_file, check_read):
    raw_trajectory = raw_data.trajectory("Sr2TiO4")
    with mock_file("trajectory", raw_trajectory) as mocks:
        check_read(Trajectory, mocks, raw_trajectory)
