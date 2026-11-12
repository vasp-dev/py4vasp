# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import h5py
import numpy as np
import pytest
from util import write_setup_file

from py4vasp import raw
from py4vasp._calculation._CONTCAR import CONTCAR
from py4vasp._calculation.structure import Structure
from py4vasp._raw import read


def test_read_structure(raw_data, tmp_path, Assert):
    raw_structure = raw_data.structure("ZnS")
    filename = tmp_path / "POSCAR"
    with open(filename, "w") as file:
        file.write(str(Structure.from_data(raw_structure)))
    structure = read.structure(filename)
    Assert.same_raw_structure(structure, raw_structure)


def test_read_contcar(raw_data, tmp_path, Assert):
    raw_structure = raw_data.structure("ZnS")
    raw_contcar = raw.CONTCAR(
        structure=raw_structure,
        system="ZnS structure",
        lattice_velocities=raw.VaspData(np.linspace(-1, 1, 9).reshape(3, 3)),
        ion_velocities=raw.VaspData(0.2 - 0.1 * raw_structure.positions),
    )
    filename = tmp_path / "POSCAR"
    with open(filename, "w") as file:
        file.write(str(CONTCAR.from_data(raw_contcar)))
    contcar = read.CONTCAR(filename)
    Assert.same_raw_contcar(contcar, raw_contcar)


@pytest.fixture
def setup_file(tmp_path):
    return lambda setup, **kwargs: write_setup_file(
        tmp_path / "vaspout.h5", setup, **kwargs
    )


def values(extension):
    return {
        section: setup_section.values
        for section, setup_section in extension.sections.items()
    }


def test_run_info_extension_reads_all_sections(setup_file):
    filename = setup_file(
        {
            "dimensions/NGX": 24,
            "dimensions/NGY": 30,
            "electronic/NELM": 60,
            "electronic/GGA": "PE",
            "dos/SIGMA": 0.05,
            "my_new_section/FOO": np.array([1.0, 2.0]),
        }
    )
    extension = read.run_info_extension(filename)
    actual = values(extension)
    assert actual.keys() == {"dimensions", "electronic", "dos", "my_new_section"}
    assert actual["dimensions"] == {"NGX": 24, "NGY": 30}
    assert actual["electronic"] == {"NELM": 60, "GGA": "PE"}
    assert actual["dos"] == {"SIGMA": 0.05}
    assert np.allclose(actual["my_new_section"]["FOO"], [1.0, 2.0])


def test_run_info_extension_flattens_subgroups(setup_file):
    filename = setup_file(
        {
            "exchange_correlation/LHFCALC": 0,
            "exchange_correlation/functional_1/NAME": "PBE",
            "exchange_correlation/functional_1/P1": 0.25,
            "exchange_correlation/functional_2/NAME": "LDA",
        }
    )
    extension = read.run_info_extension(filename)
    assert values(extension) == {
        "exchange_correlation": {
            "LHFCALC": 0,
            "functional_1/NAME": "PBE",
            "functional_1/P1": 0.25,
            "functional_2/NAME": "LDA",
        }
    }


def test_run_info_extension_keeps_units_and_description(setup_file):
    filename = setup_file(
        {
            "dos/SIGMA": (0.05, {"units": "eV", "description": "smearing width"}),
            "dos/NEDOS": (301, {"description": "number of grid points"}),
            "dos/ISMEAR": 0,
        }
    )
    extension = read.run_info_extension(filename)
    section = extension.sections["dos"]
    assert section.units == {"SIGMA": "eV"}
    assert section.descriptions == {
        "SIGMA": "smearing width",
        "NEDOS": "number of grid points",
    }


def test_run_info_extension_strips_strings(setup_file):
    filename = setup_file(
        {
            "electronic/GGA": ("PE   ", {"units": "none   ", "description": "xc  "}),
            "electronic/NAMES": np.array([b"Sr  ", b"Ti  ", b"O   "]),
        }
    )
    section = read.run_info_extension(filename).sections["electronic"]
    assert section.values["GGA"] == "PE"
    assert section.values["NAMES"].tolist() == ["Sr", "Ti", "O"]
    assert section.units == {"GGA": "none"}
    assert section.descriptions == {"GGA": "xc"}


@pytest.mark.parametrize(
    "dtype", [np.dtype("S4"), h5py.string_dtype()], ids=["fixed", "vlen"]
)
def test_run_info_extension_reads_string_array(setup_file, dtype):
    names = [["Sr", "Ti", "O"], ["La", "Al", "N"]]
    filename = setup_file(
        {"species/NAMES": np.array(names, dtype=object).astype(dtype)}
    )
    actual = read.run_info_extension(filename).sections["species"].values["NAMES"]
    assert isinstance(actual, np.ndarray)
    assert actual.shape == (2, 3)
    assert actual.tolist() == names
    assert all(isinstance(name, str) for name in actual.flat)


def test_run_info_extension_converts_bool(setup_file):
    bool_attribute = {"dtype": "bool"}
    filename = setup_file(
        {
            "electronic/LNONCOLLINEAR": (1, bool_attribute),
            "electronic/LSORBIT": (0, bool_attribute),
            "write_flags/LFLAGS": (np.array([1, 0, 1]), bool_attribute),
        }
    )
    extension = read.run_info_extension(filename)
    electronic = extension.sections["electronic"].values
    assert electronic["LNONCOLLINEAR"] is True
    assert electronic["LSORBIT"] is False
    flags = extension.sections["write_flags"].values["LFLAGS"]
    assert flags.dtype == np.bool_
    assert flags.tolist() == [True, False, True]


def test_run_info_extension_integer_without_attribute_stays_int(setup_file):
    filename = setup_file(
        {"electronic/ISPIN": 1, "electronic/NELM": (0, {"dtype": "int"})}
    )
    values = read.run_info_extension(filename).sections["electronic"].values
    assert values == {"ISPIN": 1, "NELM": 0}
    assert all(type(value) is int for value in values.values())
