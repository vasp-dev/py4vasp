# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

from py4vasp import exception
from py4vasp._calculation._stoichiometry import Stoichiometry
from py4vasp._calculation.selection import Selection
from py4vasp._util import check, convert, import_, select

ase = import_.optional("ase")
pd = import_.optional("pandas")
pdt = import_.optional("pandas.testing")


@pytest.fixture
def without_types(raw_data):
    raw_stoichiometry = raw_data.stoichiometry("Sr2TiO4 without ion types")
    return Stoichiometry.from_data(raw_stoichiometry)


class Base:
    def test_read(self):
        stoichiometry = self.stoichiometry.read(**self.ion_types)
        assert stoichiometry[select.all] == Selection(indices=slice(0, 7))
        for i, name in enumerate(self.names):
            expected = Selection(indices=slice(i, i + 1), label=name)
            assert stoichiometry[str(i + 1)] == expected
        self.check_ion_indices(stoichiometry)

    def test_to_frame(self, not_core):
        actual = self.stoichiometry.to_frame(**self.ion_types)
        ref_data = {"name": self.names, "element": self.elements}
        reference = pd.DataFrame(ref_data)
        pdt.assert_frame_equal(reference, actual)

    def test_to_mdtraj(self, not_core):
        actual, _ = self.stoichiometry.to_mdtraj(**self.ion_types).to_dataframe()
        num_atoms = self.stoichiometry.number_atoms()
        ref_data = {
            "serial": num_atoms * (None,),
            "name": self.names,
            "element": self.elements,
            "resSeq": num_atoms * (0,),
            "resName": num_atoms * ("crystal",),
            "chainID": num_atoms * (0,),
            "segmentID": num_atoms * ("",),
        }
        if "formal_charge" in actual:
            ref_data["formal_charge"] = num_atoms * (0,)
        reference = pd.DataFrame(ref_data)
        pdt.assert_frame_equal(reference, actual, check_dtype=False)

    def test_elements(self):
        assert self.stoichiometry.elements(**self.ion_types) == self.elements

    def test_ion_types(self):
        assert self.stoichiometry.ion_types(**self.ion_types) == self.unique_elements

    def test_names(self):
        actual = self.stoichiometry.names(**self.ion_types)
        assert actual == self.names

    def test_number_atoms(self):
        assert self.stoichiometry.number_atoms() == 7

    def test_from_ase(self, not_core):
        structure = ase.Atoms("".join(self.elements))
        stoichiometry = Stoichiometry.from_ase(structure)
        assert stoichiometry.elements() == self.elements
        if not hasattr(self, "overwrite_ion_types"):
            assert str(stoichiometry) == str(self.stoichiometry)

    def test_to_string(self):
        actual = self.stoichiometry.to_string(**self.ion_types)
        assert actual == self.string_format

    def test_to_poscar(self):
        poscar_string = self.stoichiometry.to_POSCAR(**self.ion_types)
        assert poscar_string == self.poscar_string

    def test_print(self, format_):
        actual, _ = format_(self.stoichiometry)
        assert actual == self.format_output

    def test_to_database(self):
        db_dict = self.stoichiometry._read_to_database()["stoichiometry:default"]
        expected_ion_types = getattr(self, "ref_ion_types", self.unique_elements)
        expected_num_ion_types = getattr(self, "ref_num_ion_types", None)
        assert (
            db_dict["ion_types"] == expected_ion_types
        ), f"ion_types mismatch: {db_dict['ion_types']} vs. {expected_ion_types}"
        assert (
            db_dict["num_ion_types"] == expected_num_ion_types
        ), f"num_ion_types mismatch: {db_dict['num_ion_types']} vs. {expected_num_ion_types}"
        if db_dict["num_ion_types"] is not None:
            for nt, rnt in zip(
                db_dict["num_ion_types"],
                db_dict["num_ion_types_primitive"],
            ):
                assert (
                    rnt <= nt
                ), f"Primitive cell has more atoms ({rnt}) than the full cell ({nt})"
            if getattr(self, "ref_num_ion_types_primitive", None) is not None:
                assert (
                    db_dict["num_ion_types_primitive"]
                    == self.ref_num_ion_types_primitive
                ), f"num_ion_types_primitive mismatch: {db_dict['num_ion_types_primitive']} vs. {self.ref_num_ion_types_primitive}"
        else:
            assert (
                db_dict["num_ion_types_primitive"] is None
            ), f"If num_ion_types is None, num_ion_types_primitive must be None as well but is {db_dict['num_ion_types_primitive']}."
        assert isinstance(db_dict["formula"], (str, type(None)))
        if getattr(self, "ref_formula", None) is not None:
            assert db_dict["formula"] == self.ref_formula
        assert isinstance(db_dict["compound"], (str, type(None)))
        if getattr(self, "ref_compound", None) is not None:
            assert db_dict["compound"] == self.ref_compound

    @property
    def unique_elements(self):
        elements = []
        for element in self.elements:
            if element not in elements:
                elements.append(element)
        return elements

    @property
    def ion_types(self):
        if hasattr(self, "overwrite_ion_types"):
            return {"ion_types": self.overwrite_ion_types}
        else:
            return {}


class TestSr2TiO4(Base):
    @pytest.fixture(autouse=True)
    def _setup(self, raw_data):
        self.stoichiometry = Stoichiometry.from_data(raw_data.stoichiometry("Sr2TiO4"))
        self.names = ["Sr_1", "Sr_2", "Ti_1", "O_1", "O_2", "O_3", "O_4"]
        self.elements = 2 * ["Sr"] + ["Ti"] + 4 * ["O"]
        self.poscar_string = "Sr Ti O\n2 1 4"
        self.string_format = "Sr2TiO4"
        self.format_output = {
            "text/plain": "Sr2TiO4",
            "text/html": "Sr<sub>2</sub>TiO<sub>4</sub>",
        }
        self.ref_ion_types = ["O", "Sr", "Ti"]
        self.ref_num_ion_types = [4, 2, 1]
        self.ref_num_ion_types_primitive = [4, 2, 1]
        self.ref_formula = "O4Sr2Ti"
        self.ref_compound = "O-Sr-Ti"

    def check_ion_indices(self, stoichiometry):
        assert stoichiometry["Sr"] == Selection(indices=slice(0, 2), label="Sr")
        assert stoichiometry["Ti"] == Selection(indices=slice(2, 3), label="Ti")
        assert stoichiometry["O"] == Selection(indices=slice(3, 7), label="O")

    def test_to_poscar_format(self):
        assert self.stoichiometry.to_POSCAR(".format.") == "Sr Ti O.format.\n2 1 4"
        with pytest.raises(exception.IncorrectUsage):
            self.stoichiometry.to_POSCAR(None)


class TestCa3AsBr3(Base):
    # test duplicate entries in POTCAR

    @pytest.fixture(autouse=True)
    def _setup(self, raw_data):
        raw_stoichiometry = raw_data.stoichiometry("Ca2AsBr-CaBr2")
        self.stoichiometry = Stoichiometry.from_data(raw_stoichiometry)
        self.names = ["Ca_1", "Ca_2", "As_1", "Br_1", "Ca_3", "Br_2", "Br_3"]
        self.elements = ["Ca", "Ca", "As", "Br", "Ca", "Br", "Br"]
        self.poscar_string = "Ca As Br Ca Br\n2 1 1 1 2"
        self.string_format = "Ca3AsBr3"
        self.format_output = {
            "text/plain": "Ca3AsBr3",
            "text/html": "Ca<sub>3</sub>AsBr<sub>3</sub>",
        }
        self.ref_ion_types = ["As", "Br", "Ca"]
        self.ref_num_ion_types = [1, 3, 3]
        self.ref_num_ion_types_primitive = [1, 3, 3]
        self.ref_formula = "AsBr3Ca3"
        self.ref_compound = "As-Br-Ca"

    def check_ion_indices(self, stoichiometry):
        assert stoichiometry["Ca"] == Selection(indices=[0, 1, 4], label="Ca")
        assert stoichiometry["As"] == Selection(indices=slice(2, 3), label="As")
        assert stoichiometry["Br"] == Selection(indices=[3, 5, 6], label="Br")


class TestBa2MnO4(Base):
    # test replacing elements via arguments

    @pytest.fixture(params=("Sr2TiO4", "Sr2TiO4 without ion types"), autouse=True)
    def _setup(self, request, raw_data):
        raw_stoichiometry = raw_data.stoichiometry(request.param)
        self.stoichiometry = Stoichiometry.from_data(raw_stoichiometry)
        self.overwrite_ion_types = ["Ba", "Mn", "O"]
        self.names = ["Ba_1", "Ba_2", "Mn_1", "O_1", "O_2", "O_3", "O_4"]
        self.elements = 2 * ["Ba"] + ["Mn"] + 4 * ["O"]
        self.poscar_string = "Ba Mn O\n2 1 4"
        self.string_format = "Ba2MnO4"
        if request.param == "Sr2TiO4":
            self.format_output = {
                "text/plain": "Sr2TiO4",
                "text/html": "Sr<sub>2</sub>TiO<sub>4</sub>",
            }
        else:
            self.format_output = {
                "text/plain": "(A)2(B)(C)4",
                "text/html": "<em>A</em><sub>2</sub><em>B</em><em>C</em><sub>4</sub>",
            }
        self.ref_ion_types = ["O", "Sr", "Ti"] if request.param == "Sr2TiO4" else None
        self.ref_num_ion_types = [4, 2, 1] if request.param == "Sr2TiO4" else None
        self.ref_num_ion_types_primitive = (
            None if not request.param == "Sr2TiO4" else [4, 2, 1]
        )
        self.ref_formula = None if not request.param == "Sr2TiO4" else "O4Sr2Ti"
        self.ref_compound = None if not request.param == "Sr2TiO4" else "O-Sr-Ti"

    def check_ion_indices(self, stoichiometry):
        assert stoichiometry["Ba"] == Selection(indices=slice(0, 2), label="Ba")
        assert stoichiometry["Mn"] == Selection(indices=slice(2, 3), label="Mn")
        assert stoichiometry["O"] == Selection(indices=slice(3, 7), label="O")


def test_poscar_string_without_types(without_types):
    assert without_types.to_POSCAR() == "2 1 4"


@pytest.mark.parametrize(
    "method", ("to_dict", "to_frame", "to_mdtraj", "names", "elements")
)
def test_ion_types_required(method, without_types, not_core):
    with pytest.raises(exception.IncorrectUsage):
        getattr(without_types, method)()


@pytest.mark.parametrize(
    "method", ("number_atoms", "__str__", "to_string", "to_POSCAR")
)
def test_ion_types_not_required(method, raw_data):
    raw_stoichiometry = raw_data.stoichiometry("Sr2TiO4 without ion types")
    stoichiometry = Stoichiometry.from_data(raw_stoichiometry)
    getattr(stoichiometry, method)()  # make sure this does not raise an error


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.stoichiometry("Sr2TiO4")
    check_factory_methods(Stoichiometry, data)
