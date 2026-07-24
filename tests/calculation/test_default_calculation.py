# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import inspect
import pathlib
from unittest.mock import patch

import pytest

from py4vasp import Calculation, calculation, demo
from py4vasp._raw.data import CalculationMetaData, _DatabaseData


def test_access_of_attributes():
    calc = Calculation.from_path(".")
    for key in filter(attribute_included, dir(calc)):
        getattr(calculation, key)


def attribute_included(attr):
    if attr.startswith("_"):  # do not include private attributes
        return False
    if attr.startswith("from"):  # do not include classmethods
        return False
    return True


@pytest.mark.skip("Input files are not included in current release.")
def test_assigning_to_input_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = "SYSTEM = demo INCAR file"
    calculation.INCAR = expected
    with open("INCAR", "r") as file:
        actual = file.read()
    assert actual == expected


def test_selections_on_empty_path(tmp_path):
    # Default (only_available=False) still returns all schema-defined quantities on empty path
    calc = Calculation.from_path(tmp_path)
    full = calc.selections()
    assert "band" in full
    assert "bandgap" in full
    assert full["band"] == ["default", "kpoints_opt", "kpoints_wan"]


def test_selections_on_demo_calculation(tmp_path):
    calc = demo.calculation(tmp_path / "demo_calculation")
    actual = calc.selections()
    # Default now returns all public quantities with schema-defined selections
    assert "band" in actual
    assert "bandgap" in actual  # included even without data
    assert "density" in actual
    assert "structure" in actual
    # the result is sorted by quantity name
    assert list(actual) == sorted(actual)


def test_selections_only_available_reports_present_data(tmp_path):
    # only_available=True reports the sources whose data is present in the output
    calc = demo.calculation(tmp_path / "demo_calculation")
    actual = calc.selections(only_available=True)
    # structure data is present under its default source
    assert "structure" in actual
    assert "default" in actual["structure"]
    # a quantity with no data in the demo is omitted entirely
    assert "born_effective_charge" not in actual


def test_selections_evaluable(tmp_path):
    calculation = demo.calculation(tmp_path / "demo_calculation")
    # selections with method parameter should work and return available sources
    viewable = calculation.selections(method="to_view")
    assert isinstance(viewable, dict)
    for quantity, sources in viewable.items():
        assert isinstance(sources, list)
        assert all(isinstance(s, str) for s in sources)


def test_selections_includes_quantities_without_data(tmp_path):
    # Default now includes all quantities; quantities without data have empty selections
    calc = demo.calculation(tmp_path / "demo_calculation")
    actual = calc.selections()
    included = (
        "bandgap",
        "born_effective_charge",
        "dielectric_function",
        "dielectric_tensor",
        "elastic_modulus",
        "internal_strain",
        "piezoelectric_tensor",
        "polarization",
    )
    for quantity in included:
        assert quantity in actual


def test_selections_filtered_by_method(tmp_path):
    calc = demo.calculation(tmp_path / "demo_calculation")
    viewable = calc.selections(method="to_view")
    full = calc.selections()
    # only quantities implementing the method are reported
    assert set(viewable) <= set(full)
    assert viewable.keys() >= {"density", "potential", "structure"}
    # quantities without a to_view method are excluded
    for quantity in ("band", "dos", "energy", "stress"):
        assert quantity not in viewable


def test_selections_with_method_on_empty_path(tmp_path):
    # Default (only_available=False) with method filter still returns quantities implementing the method
    calc = Calculation.from_path(tmp_path)
    result = calc.selections(method="to_view")
    assert "density" in result
    assert "structure" in result
    assert "band" not in result


def test_selections_with_only_available_true(tmp_path):
    calc = demo.calculation(tmp_path / "demo_calculation")
    available = calc.selections(only_available=True)
    full = calc.selections(only_available=False)
    # available quantities should be a subset of all quantities
    assert set(available) <= set(full)
    # quantities without any data should not appear in the available result
    absent_when_unavailable = {
        "bandgap",
        "born_effective_charge",
        "dielectric_tensor",
        "elastic_modulus",
        "internal_strain",
        "piezoelectric_tensor",
        "polarization",
    }
    for quantity in absent_when_unavailable:
        assert quantity not in available
        assert quantity in full
        assert "default" in full[quantity]
        assert full[quantity]


def test_selections_on_empty_path_returns_all(tmp_path):
    # Default (only_available=False) returns schema-defined selections even without data
    calc = Calculation.from_path(tmp_path)
    full = calc.selections()

    assert full["band"] == ["default", "kpoints_opt", "kpoints_wan"]
    assert "default" in full["structure"]
    assert "final" in full["structure"]
    assert "poscar" in full["structure"]
    assert full["exciton.density"] == ["default"]


def test_selections_on_empty_path_only_available_true(tmp_path):
    # With only_available=True on empty path, nothing loads
    calc = Calculation.from_path(tmp_path)
    assert calc.selections(only_available=True) == {}


def test_selections_with_method_filters_by_implementation(tmp_path):
    calc = demo.calculation(tmp_path / "demo_calculation")
    # Default: all quantities implementing to_view with schema selections
    full_view = calc.selections(method="to_view")

    assert "band" not in full_view
    assert "dos" not in full_view
    assert full_view["density"] == ["default", "tau"]
    assert "default" in full_view["structure"]


def test_all_quantities_implement_read(tmp_path):
    """Every quantity implements read(), so selections(method="read") lists them all."""
    calc = demo.calculation(tmp_path / "demo_calculation")
    assert set(calc.selections(method="read")) == set(calc.selections())


def test_selections_only_available_false_does_not_load_data(tmp_path):
    # only_available=False only inspects the schema; it must never access the files
    calc = demo.calculation(tmp_path / "demo_calculation")
    with patch("py4vasp.raw.access") as mock_access:
        assert calc.selections()
        assert calc.selections(method="to_view")
    mock_access.assert_not_called()


def test_is_available_returns_nested_dict(tmp_path):
    calc = demo.calculation(tmp_path / "demo_calculation")
    result = calc.is_available()
    assert isinstance(result, dict)
    # nested {quantity: {source: bool}}, mirroring the database layout
    assert result["structure"]["default"] is True
    assert result["structure"]["final"] is False
    for sources in result.values():
        assert isinstance(sources, dict)
        assert all(isinstance(available, bool) for available in sources.values())


def test_is_available_with_method_filters_by_implementation(tmp_path):
    calc = demo.calculation(tmp_path / "demo_calculation")
    result = calc.is_available(method="to_view")
    # only quantities implementing to_view are reported
    assert "density" in result
    assert "band" not in result


def test_is_available_and_selections_agree(tmp_path):
    # selections(only_available=True) lists exactly the sources is_available marks True
    calc = demo.calculation(tmp_path / "demo_calculation")
    availability = calc.is_available()
    selections = calc.selections(only_available=True)
    for quantity, sources in availability.items():
        available_sources = [source for source, ok in sources.items() if ok]
        if available_sources:
            assert selections[quantity] == available_sources
        else:
            assert quantity not in selections


# ---------------------------------------------------------------------------
# Calculation._to_database(): container contract and property structure.
# These exercise the aggregation in py4vasp._calculation, not any _util helper.
# ---------------------------------------------------------------------------


@pytest.fixture
def demo_db(tmp_path):
    calc = demo.calculation(tmp_path / "demo_calc")
    return calc._to_database()


def test_to_database_returns_database_data(demo_db):
    assert isinstance(demo_db, _DatabaseData)


def test_to_database_takes_no_arguments():
    """_to_database() must not accept any arguments at all (no tags, no fermi_energy)."""
    sig = inspect.signature(Calculation._to_database)
    params = [name for name in sig.parameters if name != "self"]
    assert params == []


def test_metadata_path_is_calculation_directory(demo_db):
    assert isinstance(demo_db.metadata.path, pathlib.Path)
    # path must be a directory, not a .h5 file
    assert demo_db.metadata.path.is_dir()
    assert not demo_db.metadata.path.name.endswith(".h5")


def test_metadata_schema_version_is_nonempty_string(demo_db):
    assert isinstance(demo_db.metadata.schema_version, str)
    assert demo_db.metadata.schema_version != ""


def test_properties_is_dict(demo_db):
    assert isinstance(demo_db.properties, dict)


def test_properties_has_entries(demo_db):
    assert len(demo_db.properties) > 0


def test_properties_values_are_dicts(demo_db):
    """properties is a dict of dicts: each value maps selection -> model."""
    for key, value in demo_db.properties.items():
        assert isinstance(value, dict), f"properties[{key!r}] is not a dict"
        assert len(value) > 0, f"properties[{key!r}] is empty"


def test_default_selection_is_inner_key(demo_db):
    """run_info is always available under its default selection key 'default'."""
    assert "default" in demo_db.properties["run_info"]


def test_no_leading_underscore_in_properties_keys(demo_db):
    """Private quantities like _CONTCAR must be stored under 'CONTCAR', not '_CONTCAR'."""
    for key in demo_db.properties:
        assert not key.startswith("_"), f"Key {key!r} starts with underscore"


def test_run_info_in_properties(demo_db):
    """run_info is always available and must be present in properties."""
    assert "run_info" in demo_db.properties


def test_subcomponents_not_top_level(demo_db):
    """stoichiometry and dispersion are folded into their parent quantities and
    must not appear as top-level properties."""
    assert "stoichiometry" not in demo_db.properties
    assert "dispersion" not in demo_db.properties


def test_stoichiometry_folded_into_structure(demo_db):
    """Structure models carry the folded stoichiometry fields."""
    structure_models = list(demo_db.properties["structure"].values())
    assert any(model.formula is not None for model in structure_models)
    assert any(model.ion_types is not None for model in structure_models)


def test_dispersion_folded_into_band(demo_db):
    """The band model carries the folded dispersion eigenvalue range."""
    band = demo_db.properties["band"]["default"]
    assert band.eigenvalue_min is not None
    assert band.eigenvalue_max is not None


def test_default_selection_key_has_no_suffix(demo_db):
    """Keys for the default selection must not have a '_default' suffix."""
    for key in demo_db.properties:
        assert not key.endswith("_default"), f"Key {key!r} still has '_default' suffix"


def test_non_default_selection_key_format(demo_db):
    """Selections are nested keys, not folded into the top-level quantity key."""
    for quantity, selections in demo_db.properties.items():
        assert ":" not in quantity, f"Key {quantity!r} contains a colon"
        assert "." not in quantity, f"Key {quantity!r} contains a dot"
        assert "_" != quantity[-1:], f"Key {quantity!r} ends with underscore"
        assert isinstance(selections, dict)
        for selection in selections:
            assert ":" not in selection, f"Selection {selection!r} contains a colon"


def _basic_db_checks(db, minimum_counter=1):
    assert isinstance(db, _DatabaseData)
    assert isinstance(db.metadata, CalculationMetaData)
    assert isinstance(db.metadata.path, pathlib.Path)
    assert isinstance(db.properties, dict)
    # count the non-empty inner models across all quantities and selections
    non_empty = sum(
        1
        for selections in db.properties.values()
        if isinstance(selections, dict)
        for model in selections.values()
        if model not in (None, {}, [])
    )
    assert non_empty > minimum_counter
    assert "run_info" in db.properties


@pytest.mark.parametrize(
    ["selection", "minimum_counter"],
    [(None, 5), ("collinear", 1), ("noncollinear", 5), ("spin_texture", 2)],
)
def test_to_database_on_demo_calculation(tmp_path, selection, minimum_counter):
    """Basic _to_database functionality across spin selections on a demo calculation."""
    demo_calc = demo.calculation(tmp_path / "demo_calculation", selection=selection)
    _basic_db_checks(demo_calc._to_database(), minimum_counter=minimum_counter)
