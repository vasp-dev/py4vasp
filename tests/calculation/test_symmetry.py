# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import types

import numpy as np
import pytest

from py4vasp import exception, raw
from py4vasp._calculation.symmetry import Symmetry


@pytest.fixture(params=["CoO", "AlP"])
def symmetry(raw_data, request):
    raw_symmetry = raw_data.symmetry(request.param)
    return setup_reference(raw_symmetry, request.param)


def setup_reference(raw_symmetry, name):
    symmetry = Symmetry.from_data(raw_symmetry)
    symmetry.ref = types.SimpleNamespace()
    symmetry.ref.name = name
    symmetry.ref.raw = raw_symmetry
    return symmetry


def test_read(symmetry, Assert):
    raw = symmetry.ref.raw
    actual = symmetry.read()
    Assert.allclose(actual["rotations"], np.array(raw.rotations))
    Assert.allclose(actual["reciprocal_rotations"], np.array(raw.reciprocal_rotations))
    Assert.allclose(actual["translations"], np.array(raw.translations))
    # Fortran 1-based indices in the file are converted to 0-based for Python
    Assert.allclose(actual["inverse_operations"], np.array(raw.inverse_operations) - 1)
    Assert.allclose(actual["atom_permutations"], np.array(raw.atom_permutations) - 1)
    Assert.allclose(
        actual["primitive_lattice_vectors"], np.array(raw.primitive_lattice_vectors)
    )
    Assert.allclose(
        actual["primitive_translations"], np.array(raw.primitive_translations)
    )
    assert actual["number_of_operations"] == raw.number_of_operations
    assert actual["number_of_primitive_cells"] == raw.number_of_primitive_cells
    assert actual["isym"] == raw.isym
    if symmetry.ref.name == "CoO":
        Assert.allclose(actual["spin_flips"], np.array(raw.spin_flips))
    else:
        assert "spin_flips" not in actual


def test_to_dict_is_alias_of_read(symmetry, Assert):
    from_read = symmetry.read()
    from_dict = symmetry.to_dict()
    assert from_read.keys() == from_dict.keys()
    for key in from_read:
        if from_read[key] is None:
            assert from_dict[key] is None
        else:
            Assert.allclose(from_dict[key], from_read[key])


EXPECTED_SPACE_GROUP = {
    "CoO": dict(
        number=216,
        international_symbol="F-43m",
        point_group="-43m",
        crystal_system="cubic",
        point_group_schoenflies="Td",
        bravais_lattice="cF",
        pearson_symbol="cF8",
    ),
    "AlP": dict(
        number=38,
        international_symbol="Amm2",
        point_group="mm2",
        crystal_system="orthorhombic",
        point_group_schoenflies="C2v",
        bravais_lattice="oS",
        pearson_symbol="oS4",
    ),
}


def test_bravais_lattice(symmetry):
    pytest.importorskip("spglib")
    expected = EXPECTED_SPACE_GROUP[symmetry.ref.name]["bravais_lattice"]
    assert symmetry.bravais_lattice() == expected


def test_point_group_schoenflies(symmetry):
    pytest.importorskip("spglib")
    expected = EXPECTED_SPACE_GROUP[symmetry.ref.name]["point_group_schoenflies"]
    assert symmetry.point_group_schoenflies() == expected


def test_pearson_symbol(symmetry):
    pytest.importorskip("spglib")
    expected = EXPECTED_SPACE_GROUP[symmetry.ref.name]["pearson_symbol"]
    assert symmetry.pearson_symbol() == expected


def test_space_group(symmetry):
    pytest.importorskip("spglib")
    from py4vasp._calculation.symmetry import SpaceGroup

    expected = EXPECTED_SPACE_GROUP[symmetry.ref.name]
    actual = symmetry.space_group()
    assert isinstance(actual, SpaceGroup)
    assert actual.number == expected["number"]
    assert actual.international_symbol == expected["international_symbol"]
    assert actual.point_group == expected["point_group"]
    assert actual.crystal_system == expected["crystal_system"]
    assert actual.is_symmorphic is True


def test_space_group_without_spglib(symmetry, monkeypatch):
    from py4vasp._calculation import symmetry as symmetry_module
    from py4vasp._util import import_

    placeholder = import_._ModulePlaceholder("spglib")
    monkeypatch.setattr(symmetry_module, "spglib", placeholder)
    with pytest.raises(exception.ModuleNotInstalled):
        symmetry.space_group()


def test_has_inversion_symmetry_absent(symmetry):
    assert symmetry.has_inversion_symmetry() is False


def test_has_inversion_symmetry_present(raw_data):
    raw_symmetry = raw_data.symmetry("AlP")
    rotations = np.array(raw_symmetry.rotations)
    rotations[1] = -np.eye(3, dtype=int)
    modified = dataclasses.replace(raw_symmetry, rotations=raw.VaspData(rotations))
    assert Symmetry.from_data(modified).has_inversion_symmetry() is True


NUMBER_OPERATIONS = {"CoO": 24, "AlP": 4}
NUMBER_PRIMITIVE_CELLS = {"CoO": 1, "AlP": 2}


def test_print(symmetry, format_):
    pytest.importorskip("spglib")
    name = symmetry.ref.name
    space_group = EXPECTED_SPACE_GROUP[name]
    reference = f"""\
symmetry group with {NUMBER_OPERATIONS[name]} operations:
    space group: {space_group['international_symbol']} ({space_group['number']})
    crystal system: {space_group['crystal_system']}
    inversion symmetry: no
    primitive cells: {NUMBER_PRIMITIVE_CELLS[name]}
    ISYM: 2"""
    actual, _ = format_(symmetry)
    assert actual == {"text/plain": reference}


def test_print_without_spglib(symmetry, format_, monkeypatch):
    from py4vasp._calculation import symmetry as symmetry_module
    from py4vasp._util import import_

    monkeypatch.setattr(symmetry_module, "spglib", import_._ModulePlaceholder("spglib"))
    name = symmetry.ref.name
    reference = f"""\
symmetry group with {NUMBER_OPERATIONS[name]} operations:
    space group: not available (requires spglib)
    inversion symmetry: no
    primitive cells: {NUMBER_PRIMITIVE_CELLS[name]}
    ISYM: 2"""
    actual, _ = format_(symmetry)
    assert actual == {"text/plain": reference}


def test_to_database(symmetry, raw_data):
    pytest.importorskip("spglib")
    from py4vasp._calculation.symmetry import SymmetryHandler
    from py4vasp._raw.data_db import Symmetry_DB

    name = symmetry.ref.name
    handler = SymmetryHandler.from_data(raw_data.symmetry(name))
    actual = handler.to_database()
    assert isinstance(actual, Symmetry_DB)
    space_group = EXPECTED_SPACE_GROUP[name]
    assert actual.space_group == space_group["number"]
    assert actual.space_group_symbol == space_group["international_symbol"]
    assert actual.crystal_system == space_group["crystal_system"]
    assert actual.point_group_schoenflies == space_group["point_group_schoenflies"]
    assert actual.bravais_lattice == space_group["bravais_lattice"]
    assert actual.pearson_symbol == space_group["pearson_symbol"]
    assert actual.has_inversion_symmetry is False
    assert actual.number_of_operations == NUMBER_OPERATIONS[name]
    assert actual.number_of_primitive_cells == NUMBER_PRIMITIVE_CELLS[name]
    assert actual.is_symmorphic is True


def test_to_database_without_spglib(symmetry, raw_data, monkeypatch):
    from py4vasp._calculation import symmetry as symmetry_module
    from py4vasp._calculation.symmetry import SymmetryHandler
    from py4vasp._util import import_

    monkeypatch.setattr(symmetry_module, "spglib", import_._ModulePlaceholder("spglib"))
    name = symmetry.ref.name
    actual = SymmetryHandler.from_data(raw_data.symmetry(name)).to_database()
    assert actual.space_group is None
    assert actual.space_group_symbol is None
    assert actual.crystal_system is None
    assert actual.point_group_schoenflies is None
    assert actual.bravais_lattice is None
    assert actual.pearson_symbol is None
    assert actual.has_inversion_symmetry is False
    assert actual.number_of_operations == NUMBER_OPERATIONS[name]
    assert actual.number_of_primitive_cells == NUMBER_PRIMITIVE_CELLS[name]
    assert actual.is_symmorphic is True


def test_to_database_dispatcher(symmetry):
    pytest.importorskip("spglib")
    from py4vasp._calculation.symmetry import SymmetryHandler

    result = symmetry._to_database()
    assert set(result) == {"symmetry"}
    assert set(result["symmetry"]) == {"default"}
    expected = SymmetryHandler.from_data(symmetry.ref.raw).to_database()
    assert result["symmetry"]["default"] == expected


def test_factory_methods(raw_data, check_factory_methods):
    raw_symmetry = raw_data.symmetry("CoO")
    check_factory_methods(Symmetry, raw_symmetry)
