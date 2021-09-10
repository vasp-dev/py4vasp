from py4vasp.data import Topology
from py4vasp.data._selection import Selection
import py4vasp.exceptions as exception
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def Sr2TiO4(raw_data):
    return Topology(raw_data.topology("Sr2TiO4"))


def test_read(Sr2TiO4):
    topology = Sr2TiO4.read()
    assert topology["Sr"] == Selection(indices=slice(0, 2), label="Sr")
    assert topology["Ti"] == Selection(indices=slice(2, 3), label="Ti")
    assert topology["O"] == Selection(indices=slice(3, 7), label="O")
    assert topology["1"] == Selection(indices=slice(0, 1), label="Sr_1")
    assert topology["2"] == Selection(indices=slice(1, 2), label="Sr_2")
    assert topology["3"] == Selection(indices=slice(2, 3), label="Ti_1")
    assert topology["4"] == Selection(indices=slice(3, 4), label="O_1")
    assert topology["5"] == Selection(indices=slice(4, 5), label="O_2")
    assert topology["6"] == Selection(indices=slice(5, 6), label="O_3")
    assert topology["7"] == Selection(indices=slice(6, 7), label="O_4")
    assert topology["*"] == Selection(indices=slice(7))


def test_to_frame(Sr2TiO4):
    actual = Sr2TiO4.to_frame()
    ref_data = {
        "name": ("Sr_1", "Sr_2", "Ti_1", "O_1", "O_2", "O_3", "O_4"),
        "element": 2 * ("Sr",) + ("Ti",) + 4 * ("O",),
    }
    reference = pd.DataFrame(ref_data)
    assert reference.equals(actual)


def test_to_mdtraj(Sr2TiO4):
    actual, _ = Sr2TiO4.to_mdtraj().to_dataframe()
    num_atoms = Sr2TiO4.number_atoms()
    ref_data = {
        "serial": num_atoms * (None,),
        "name": ("Sr_1", "Sr_2", "Ti_1", "O_1", "O_2", "O_3", "O_4"),
        "element": 2 * ("Sr",) + ("Ti",) + 4 * ("O",),
        "resSeq": num_atoms * (0,),
        "resName": num_atoms * ("crystal",),
        "chainID": num_atoms * (0,),
        "segmentID": num_atoms * ("",),
    }
    reference = pd.DataFrame(ref_data)
    assert reference.equals(actual)


def test_to_poscar(Sr2TiO4):
    assert Sr2TiO4.to_poscar() == "Sr Ti O\n2 1 4"
    assert Sr2TiO4.to_poscar(".format.") == "Sr Ti O.format.\n2 1 4"
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.to_poscar(None)


def test_elements(Sr2TiO4):
    assert Sr2TiO4.elements() == ["Sr", "Sr", "Ti", "O", "O", "O", "O"]


def test_ion_types(Sr2TiO4):
    assert Sr2TiO4.ion_types() == ["Sr", "Ti", "O"]


def test_names(Sr2TiO4):
    assert Sr2TiO4.names() == ["Sr_1", "Sr_2", "Ti_1", "O_1", "O_2", "O_3", "O_4"]


def test_number_atoms(Sr2TiO4):
    assert Sr2TiO4.number_atoms() == 7


def test_print(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4)
    reference = {"text/plain": "Sr2TiO4", "text/html": "Sr<sub>2</sub>TiO<sub>4</sub>"}
    assert actual == reference


def test_descriptor(Sr2TiO4, check_descriptors):
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_to_frame": ["to_frame"],
        "_to_poscar": ["to_poscar"],
        "_to_mdtraj": ["to_mdtraj"],
        "_elements": ["elements"],
        "_ion_types": ["ion_types"],
        "_names": ["names"],
        "_number_atoms": ["number_atoms"],
    }
    check_descriptors(Sr2TiO4, descriptors)


# def test_from_file(raw_topology, mock_file, check_read):
#     with mock_file("topology", raw_topology) as mocks:
#         check_read(Topology, mocks, raw_topology)
