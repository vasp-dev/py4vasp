# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Every name in vasp.backend wraps a py4vasp internal, so it can break silently.

These tests are the contract: they fail when py4vasp renames or removes something the
in-house tools depend on, instead of leaving that discovery to the tools themselves.
"""

import pytest
from vasp import backend

import py4vasp
from py4vasp import demo, exception
from py4vasp._calculation import GROUPS


@pytest.fixture
def calculation(tmp_path):
    return demo.calculation(tmp_path / "database_example")


def test_every_exported_name_resolves():
    for name in backend.__all__:
        assert hasattr(backend, name), name


def test_exports_are_sorted_and_public():
    assert backend.__all__ == sorted(backend.__all__)
    assert not any(name.startswith("_") for name in backend.__all__)


def test_version_agrees_with_py4vasp():
    assert backend.__version__ == py4vasp.__version__


def test_py4vasp_still_provides_the_private_interface():
    """The single most important assumption of this package."""
    assert hasattr(py4vasp.Calculation, "_to_database")


def test_to_database_of_calculation(calculation):
    assert backend.to_database(calculation) == calculation._to_database()


def test_to_database_returns_metadata_and_properties(calculation):
    data = backend.to_database(calculation)
    assert isinstance(data, backend.DatabaseData)
    assert isinstance(data.metadata, backend.CalculationMetaData)
    assert data.metadata.schema_version == backend.schema_version()
    assert "structure" in data.properties


def test_to_database_of_single_quantity(calculation):
    quantity = calculation.structure
    assert backend.to_database(quantity) == quantity._to_database()
    assert sorted(backend.to_database(quantity)) == ["structure"]


@pytest.mark.parametrize("name", ("density", "neighbor_list", "system"))
def test_to_database_of_quantity_without_database_data(calculation, name):
    """py4vasp stores nothing for these, so extracting them is empty, not an error."""
    quantity = getattr(calculation, name)
    assert not hasattr(quantity, "_to_database")
    assert backend.to_database(quantity) == {}


@pytest.mark.parametrize("name", sorted(GROUPS))
def test_to_database_of_group_points_at_its_members(calculation, name):
    """Returning {} for a group would silently drop the data of all its members."""
    group = getattr(calculation, name)
    with pytest.raises(exception.IncorrectUsage) as failure:
        backend.to_database(group)
    for member in GROUPS[name]:
        assert member in str(failure.value)


@pytest.mark.parametrize("source", ("not a calculation", 42, None))
def test_to_database_raises_for_unsupported_argument(source):
    with pytest.raises(exception.IncorrectUsage):
        backend.to_database(source)


def test_to_database_error_does_not_leak_the_calculation_path(calculation):
    with pytest.raises(exception.IncorrectUsage) as failure:
        backend.to_database(calculation.phonon)
    assert str(calculation.path()) not in str(failure.value)


def test_schema_version_round_trips():
    series, counter = backend.parse_schema_version(backend.schema_version())
    assert series == ".".join(py4vasp.__version__.split(".")[:2])
    assert counter >= 0


@pytest.mark.parametrize("version", ("0.11", "0.11+db.5"))
def test_parse_schema_version(version):
    assert backend.parse_schema_version(version)[0] == "0.11"


@pytest.mark.parametrize("version", ("bogus", "0.11+db", "", "0.11.3"))
def test_parse_schema_version_reports_a_py4vasp_error(version):
    """py4vasp raises a bare ValueError here; consumers only catch Py4VaspError."""
    with pytest.raises(exception.IncorrectUsage):
        backend.parse_schema_version(version)


def test_combine_database_dicts_merges_recursively():
    combined = backend.combine_database_dicts({"a": {"b": 1}}, {"a": {"c": 2}})
    assert combined == {"a": {"b": 1, "c": 2}}


def test_combine_database_dicts_prefers_the_last_argument():
    assert backend.combine_database_dicts({"a": 1}, {"a": 2}) == {"a": 2}


def test_combine_database_dicts_leaves_its_arguments_untouched():
    """py4vasp merges in place and aliases nested dicts of its arguments."""
    first = {"a": {"b": 1}}
    second = {"a": {"c": 2}}
    combined = backend.combine_database_dicts(first, second)
    assert first == {"a": {"b": 1}}
    assert second == {"a": {"c": 2}}
    assert combined["a"] is not first["a"]
    assert combined["a"] is not second["a"]


def test_combine_database_dicts_reports_a_merge_conflict():
    """py4vasp raises _Py4VaspInternalError, which is not a Py4VaspError."""
    with pytest.raises(exception.DataMismatch):
        backend.combine_database_dicts({"a": {"b": 1}}, {"a": 2})


@pytest.mark.parametrize("source", ("string", 42, None, [{"a": 1}]))
def test_combine_database_dicts_rejects_a_non_dictionary(source):
    with pytest.raises(exception.IncorrectUsage):
        backend.combine_database_dicts({"a": 1}, source)


def test_all_database_keys_reports_models():
    models, keys = backend.all_database_keys()
    assert keys["structure"] == "StructureModel"
    assert dict(models["StructureModel"])["cell_volume"] == "Optional[float]"


def test_all_database_keys_covers_the_extracted_quantities(calculation):
    """Mind the two spellings: to_database flattens 'phonon.band' to 'phonon_band'."""
    _, keys = backend.all_database_keys()
    dotted = {
        _flat_name(group, member): f"{group}.{member}"
        for group in GROUPS
        for member in GROUPS[group]
    }
    for quantity in backend.to_database(calculation).properties:
        assert dotted.get(quantity, quantity) in keys, quantity


def _flat_name(group, member):
    return f"{group}_{member}"
