from py4vasp.data import Topology, _util
import py4vasp.raw as raw
import pytest
import numpy as np
import pandas as pd

Selection = _util.Selection


@pytest.fixture
def raw_topology():
    topology = raw.Topology(
        number_ion_types=np.array((2, 1, 4)),
        ion_types=np.array(("Sr", "Ti", "O "), dtype="S"),
    )
    topology.names = ["Sr_1", "Sr_2", "Ti_1", "O_1", "O_2", "O_3", "O_4"]
    topology.elements = ["Sr", "Sr", "Ti", "O", "O", "O", "O"]
    return topology


def test_raw_topology(raw_topology):
    index = np.cumsum(raw_topology.number_ion_types)
    topology = Topology(raw_topology).read()
    assert topology["Sr"] == Selection(indices=range(index[0]), label="Sr")
    assert topology["Ti"] == Selection(indices=range(index[0], index[1]), label="Ti")
    assert topology["O"] == Selection(indices=range(index[1], index[2]), label="O")
    assert topology["1"] == Selection(indices=(0,), label="Sr_1")
    assert topology["2"] == Selection(indices=(1,), label="Sr_2")
    assert topology["3"] == Selection(indices=(2,), label="Ti_1")
    assert topology["4"] == Selection(indices=(3,), label="O_1")
    assert topology["5"] == Selection(indices=(4,), label="O_2")
    assert topology["6"] == Selection(indices=(5,), label="O_3")
    assert topology["7"] == Selection(indices=(6,), label="O_4")
    assert topology["*"] == Selection(indices=range(index[-1]))


def test_atom_labels(raw_topology):
    topology = Topology(raw_topology)
    assert topology.names() == raw_topology.names
    assert topology.elements() == raw_topology.elements


def test_from_file(raw_topology, mock_file, check_read):
    with mock_file("topology", raw_topology) as mocks:
        check_read(Topology, mocks, raw_topology)


def test_to_frame(raw_topology):
    actual = Topology(raw_topology).to_frame()
    ref_data = {
        "name": ("Sr_1", "Sr_2", "Ti_1", "O_1", "O_2", "O_3", "O_4"),
        "element": 2 * ("Sr",) + ("Ti",) + 4 * ("O",),
    }
    ref = pd.DataFrame(ref_data)
    assert ref.equals(actual)


def test_to_mdtraj(raw_topology):
    topology = Topology(raw_topology).to_mdtraj()
    actual, _ = topology.to_dataframe()
    num_atoms = np.sum(raw_topology.number_ion_types)
    ref_data = {
        "serial": num_atoms * (None,),
        "name": ("Sr_1", "Sr_2", "Ti_1", "O_1", "O_2", "O_3", "O_4"),
        "element": 2 * ("Sr",) + ("Ti",) + 4 * ("O",),
        "resSeq": num_atoms * (0,),
        "resName": num_atoms * ("crystal",),
        "chainID": num_atoms * (0,),
        "segmentID": num_atoms * ("",),
    }
    ref = pd.DataFrame(ref_data)
    assert ref.equals(actual)
