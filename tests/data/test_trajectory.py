from py4vasp.data import Trajectory, Topology
from .test_topology import raw_topology
import py4vasp.raw as raw
import pytest
import numpy as np


num_atoms = 7
num_steps = 2
pm_to_A = 1.0 / Trajectory.A_to_pm


@pytest.fixture
def raw_trajectory(raw_topology):
    shape_pos = (num_steps, num_atoms, 3)
    return raw.Trajectory(
        topology=raw_topology,
        positions=(np.arange(np.prod(shape_pos)) + 1).reshape(shape_pos),
        lattice_vectors=np.array(num_steps * [np.eye(3)]),
    )


def test_read_trajectory(raw_trajectory, Assert):
    trajectory = Trajectory(raw_trajectory).read()
    assert trajectory["names"] == raw_trajectory.topology.names
    assert trajectory["elements"] == raw_trajectory.topology.elements
    Assert.allclose(trajectory["positions"], raw_trajectory.positions)
    Assert.allclose(trajectory["lattice_vectors"], raw_trajectory.lattice_vectors)


def test_from_file(raw_trajectory, mock_file, check_read):
    with mock_file("trajectory", raw_trajectory) as mocks:
        check_read(Trajectory, mocks, raw_trajectory)


def test_to_mdtraj(raw_trajectory, Assert):
    trajectory = Trajectory(raw_trajectory).to_mdtraj()
    assert trajectory.n_frames == num_steps
    assert trajectory.n_atoms == num_atoms
    Assert.allclose(trajectory.xyz * pm_to_A, raw_trajectory.positions)
    test_cells = trajectory.unitcell_vectors * pm_to_A
    Assert.allclose(test_cells, raw_trajectory.lattice_vectors)


def test_triclinic_cell(raw_trajectory, Assert):
    unit_cell = (np.arange(9) ** 2).reshape(3, 3)
    inv_cell = np.linalg.inv(unit_cell)
    triclinic_cell = raw.Trajectory(
        topology=raw_trajectory.topology,
        lattice_vectors=np.array(num_steps * [unit_cell]),
        positions=raw_trajectory.positions @ inv_cell,
    )
    trajectory = Trajectory(triclinic_cell)
    test_cells = trajectory.read()["lattice_vectors"]
    Assert.allclose(test_cells, triclinic_cell.lattice_vectors)
    trajectory = trajectory.to_mdtraj()
    Assert.allclose(trajectory.xyz * pm_to_A, raw_trajectory.positions)
    metric = lambda cell: cell @ cell.T
    test_cell = trajectory.unitcell_vectors[0] * pm_to_A
    Assert.allclose(metric(test_cell), metric(unit_cell))
