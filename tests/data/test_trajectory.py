from py4vasp.data import Trajectory, Topology, _util
from py4vasp.raw import RawTrajectory, RawTopology, RawVersion
from .test_topology import raw_topology
from . import current_vasp_version
from unittest.mock import patch
import py4vasp.exceptions as exception
import py4vasp.data as data
import pytest
import numpy as np


num_atoms = 7
num_steps = 2
pm_to_A = 1.0 / Trajectory.A_to_pm


@pytest.fixture
def raw_trajectory(raw_topology):
    shape_pos = (num_steps, num_atoms, 3)
    return RawTrajectory(
        version=current_vasp_version,
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


def test_plot(raw_trajectory):
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    with cm_init as init, cm_cell as cell:
        trajectory = Trajectory(raw_trajectory)
        trajectory.plot()
        init.assert_called_once()
        cell.assert_called_once()


def test_triclinic_cell(raw_trajectory, Assert):
    unit_cell = (np.arange(9) ** 2).reshape(3, 3)
    inv_cell = np.linalg.inv(unit_cell)
    triclinic_cell = RawTrajectory(
        version=current_vasp_version,
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


def test_to_structure(raw_trajectory, Assert):
    structure = Trajectory(raw_trajectory).to_structure(0).read()
    ref_elements = Topology(raw_trajectory.topology).elements()
    assert structure["elements"] == ref_elements
    Assert.allclose(structure["lattice_vectors"], raw_trajectory.lattice_vectors[0])
    Assert.allclose(structure["positions"], raw_trajectory.positions[0])


def test_incorrect_step(raw_trajectory):
    trajectory = Trajectory(raw_trajectory)
    with pytest.raises(exception.IncorrectUsage):
        trajectory.to_structure(100)
    with pytest.raises(exception.IncorrectUsage):
        trajectory.to_structure([0, 1])


def test_print(raw_trajectory):
    actual, _ = _util.format_(Trajectory(raw_trajectory))
    ref_plain = """
current structure of 2 step trajectory
1.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
Sr Ti O
2 1 4
Direct
22 23 24
25 26 27
28 29 30
31 32 33
34 35 36
37 38 39
40 41 42
    """.strip()
    ref_html = """
current structure of 2 step trajectory<br>
1.0<br>
<table>
<tr><td>1.0</td><td>0.0</td><td>0.0</td></tr>
<tr><td>0.0</td><td>1.0</td><td>0.0</td></tr>
<tr><td>0.0</td><td>0.0</td><td>1.0</td></tr>
</table>
Sr Ti O<br>
2 1 4<br>
Direct<br>
<table>
<tr><td>22</td><td>23</td><td>24</td></tr>
<tr><td>25</td><td>26</td><td>27</td></tr>
<tr><td>28</td><td>29</td><td>30</td></tr>
<tr><td>31</td><td>32</td><td>33</td></tr>
<tr><td>34</td><td>35</td><td>36</td></tr>
<tr><td>37</td><td>38</td><td>39</td></tr>
<tr><td>40</td><td>41</td><td>42</td></tr>
</table>
    """.strip()
    assert actual == {"text/plain": ref_plain, "text/html": ref_html}


def test_version(raw_trajectory):
    raw_trajectory.version = RawVersion(_util._minimal_vasp_version.major - 1)
    with pytest.raises(exception.OutdatedVaspVersion):
        Trajectory(raw_trajectory).read()


def test_descriptor(raw_trajectory, check_descriptors):
    trajectory = Trajectory(raw_trajectory)
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_to_viewer3d": ["to_viewer3d", "plot"],
        "_to_structure": ["to_structure"],
        "_to_mdtraj": ["to_mdtraj"],
    }
    check_descriptors(trajectory, descriptors)
