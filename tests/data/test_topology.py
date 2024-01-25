# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

from py4vasp import exception
from py4vasp.calculation._selection import Selection
from py4vasp._util import import_, select
from py4vasp.data import Topology

ase = import_.optional("ase")
pd = import_.optional("pandas")


class Base:
    def test_read(self):
        topology = self.topology.read()
        assert topology[select.all] == Selection(indices=slice(0, 7))
        for i, name in enumerate(self.names):
            expected = Selection(indices=slice(i, i + 1), label=name)
            assert topology[str(i + 1)] == expected
        self.check_ion_indices(topology)

    def test_to_frame(self, not_core):
        actual = self.topology.to_frame()
        ref_data = {"name": self.names, "element": self.elements}
        reference = pd.DataFrame(ref_data)
        assert reference.equals(actual)

    def test_to_mdtraj(self, not_core):
        actual, _ = self.topology.to_mdtraj().to_dataframe()
        num_atoms = self.topology.number_atoms()
        ref_data = {
            "serial": num_atoms * (None,),
            "name": self.names,
            "element": self.elements,
            "resSeq": num_atoms * (0,),
            "resName": num_atoms * ("crystal",),
            "chainID": num_atoms * (0,),
            "segmentID": num_atoms * ("",),
        }
        reference = pd.DataFrame(ref_data)
        assert reference.equals(actual)

    def test_elements(self):
        assert self.topology.elements() == self.elements

    def test_ion_types(self):
        assert self.topology.ion_types() == self.unique_elements

    def test_names(self):
        actual = self.topology.names()
        assert actual == self.names

    def test_number_atoms(self):
        assert self.topology.number_atoms() == 7

    def test_from_ase(self, not_core):
        structure = ase.Atoms("".join(self.elements))
        topology = Topology.from_ase(structure)
        assert topology.elements() == self.elements
        assert str(topology) == str(self.topology)

    @property
    def unique_elements(self):
        elements = []
        for element in self.elements:
            if element not in elements:
                elements.append(element)
        return elements


class TestSr2TiO4(Base):
    @pytest.fixture(autouse=True)
    def _setup(self, raw_data):
        self.topology = Topology.from_data(raw_data.topology("Sr2TiO4"))
        self.names = ["Sr_1", "Sr_2", "Ti_1", "O_1", "O_2", "O_3", "O_4"]
        self.elements = 2 * ["Sr"] + ["Ti"] + 4 * ["O"]

    def check_ion_indices(self, topology):
        assert topology["Sr"] == Selection(indices=slice(0, 2), label="Sr")
        assert topology["Ti"] == Selection(indices=slice(2, 3), label="Ti")
        assert topology["O"] == Selection(indices=slice(3, 7), label="O")

    def test_to_poscar(self):
        assert self.topology.to_POSCAR() == "Sr Ti O\n2 1 4"
        assert self.topology.to_POSCAR(".format.") == "Sr Ti O.format.\n2 1 4"
        with pytest.raises(exception.IncorrectUsage):
            self.topology.to_POSCAR(None)

    def test_print(self, format_):
        actual, _ = format_(self.topology)
        reference = {
            "text/plain": "Sr2TiO4",
            "text/html": "Sr<sub>2</sub>TiO<sub>4</sub>",
        }
        assert actual == reference


class TestCa3AsBr3(Base):
    # test duplicate entries in POTCAR

    @pytest.fixture(autouse=True)
    def _setup(self, raw_data):
        self.topology = Topology.from_data(raw_data.topology("Ca2AsBr-CaBr2"))
        self.names = ["Ca_1", "Ca_2", "As_1", "Br_1", "Ca_3", "Br_2", "Br_3"]
        self.elements = ["Ca", "Ca", "As", "Br", "Ca", "Br", "Br"]

    def check_ion_indices(self, topology):
        assert topology["Ca"] == Selection(indices=[0, 1, 4], label="Ca")
        assert topology["As"] == Selection(indices=slice(2, 3), label="As")
        assert topology["Br"] == Selection(indices=[3, 5, 6], label="Br")

    def test_to_poscar(self):
        assert self.topology.to_POSCAR() == "Ca As Br Ca Br\n2 1 1 1 2"

    def test_print(self, format_):
        actual, _ = format_(self.topology)
        reference = {
            "text/plain": "Ca3AsBr3",
            "text/html": "Ca<sub>3</sub>AsBr<sub>3</sub>",
        }
        assert actual == reference


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.topology("Sr2TiO4")
    check_factory_methods(Topology, data)
