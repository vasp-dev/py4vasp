# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pandas as pd
import pytest

from py4vasp import exception
from py4vasp._data.selection import Selection
from py4vasp._util import select
from py4vasp.data import Topology


@pytest.fixture
def Sr2TiO4(raw_data):
    return Topology.from_data(raw_data.topology("Sr2TiO4"))


def test_Sr2TiO4_read(Sr2TiO4):
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
    assert topology[select.all] == Selection(indices=slice(0, 7))


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
    assert Sr2TiO4.to_POSCAR() == "Sr Ti O\n2 1 4"
    assert Sr2TiO4.to_POSCAR(".format.") == "Sr Ti O.format.\n2 1 4"
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.to_POSCAR(None)


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


class TestCa3AsBr3:
    # test duplicate entries in POTCAR

    @pytest.fixture(autouse=True)
    def _setup(self, raw_data):
        self.topology = Topology.from_data(raw_data.topology("Ca2AsBr-CaBr2"))

    def test_read(self):
        topology = self.topology.read()
        assert topology["Ca"] == Selection(indices=[0, 1, 4], label="Ca")
        assert topology["As"] == Selection(indices=slice(2, 3), label="As")
        assert topology["Br"] == Selection(indices=[3, 5, 6], label="Br")
        assert topology["1"] == Selection(indices=slice(0, 1), label="Ca_1")
        assert topology["2"] == Selection(indices=slice(1, 2), label="Ca_2")
        assert topology["3"] == Selection(indices=slice(2, 3), label="As_1")
        assert topology["4"] == Selection(indices=slice(3, 4), label="Br_1")
        assert topology["5"] == Selection(indices=slice(4, 5), label="Ca_3")
        assert topology["6"] == Selection(indices=slice(5, 6), label="Br_2")
        assert topology["7"] == Selection(indices=slice(6, 7), label="Br_3")
        assert topology[select.all] == Selection(indices=slice(0, 7))

    def test_to_frame(self):
        actual = self.topology.to_frame()
        ref_data = {
            "name": ("Ca_1", "Ca_2", "As_1", "Br_1", "Ca_3", "Br_2", "Br_3"),
            "element": ("Ca", "Ca", "As", "Br", "Ca", "Br", "Br"),
        }
        reference = pd.DataFrame(ref_data)
        assert reference.equals(actual)

    def test_to_mdtraj(self):
        actual, _ = self.topology.to_mdtraj().to_dataframe()
        num_atoms = self.topology.number_atoms()
        ref_data = {
            "serial": num_atoms * (None,),
            "name": ("Ca_1", "Ca_2", "As_1", "Br_1", "Ca_3", "Br_2", "Br_3"),
            "element": ("Ca", "Ca", "As", "Br", "Ca", "Br", "Br"),
            "resSeq": num_atoms * (0,),
            "resName": num_atoms * ("crystal",),
            "chainID": num_atoms * (0,),
            "segmentID": num_atoms * ("",),
        }
        reference = pd.DataFrame(ref_data)
        assert reference.equals(actual)

    def test_to_poscar(self):
        assert self.topology.to_POSCAR() == "Ca As Br Ca Br\n2 1 1 1 2"

    def test_elements(self):
        assert self.topology.elements() == ["Ca", "Ca", "As", "Br", "Ca", "Br", "Br"]

    def test_ion_types(self):
        assert self.topology.ion_types() == ["Ca", "As", "Br"]

    def test_names(self):
        actual = self.topology.names()
        assert actual == ["Ca_1", "Ca_2", "As_1", "Br_1", "Ca_3", "Br_2", "Br_3"]

    def test_number_atoms(self):
        assert self.topology.number_atoms() == 7


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.topology("Sr2TiO4")
    check_factory_methods(Topology, data)
