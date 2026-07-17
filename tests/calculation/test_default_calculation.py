# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import contextlib
import inspect
import pathlib

import pytest

from py4vasp import Calculation, calculation, demo
from py4vasp._raw.data import CalculationMetaData, _DatabaseData
from py4vasp._raw.schema import DEFAULT_SELECTION
from py4vasp._util import loadable


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


def test_selections_loadable_excludes_selections_that_do_not_load(tmp_path):
    # Use only_available=True to get only loadable quantities
    calc = demo.calculation(tmp_path / "demo_calculation")
    actual = calc.selections(only_available=True)
    # current_density cannot be read without specifying a cut plane -> excluded
    assert "current_density" not in actual
    # a non-default source of a not-yet-migrated quantity cannot be addressed
    assert actual["structure"] == ["default"]


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
    loadable = calc.selections(only_available=True)
    full = calc.selections(only_available=False)
    # loadable quantities should be a subset of all quantities
    assert set(loadable) <= set(full)
    # quantities without loadable data should not appear in loadable result
    # (the demo does generate a dielectric_function, so it is loadable and excluded here)
    absent_in_loadable = {
        "bandgap",
        "born_effective_charge",
        "dielectric_tensor",
        "elastic_modulus",
        "internal_strain",
        "piezoelectric_tensor",
        "polarization",
    }
    for quantity in absent_in_loadable:
        assert quantity not in loadable
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
    """Verify all quantities implement the read() method."""
    calc = demo.calculation(tmp_path / "demo_calculation")
    all_quantities = calc.selections(only_available=False)
    for quantity_name in all_quantities:
        assert loadable.implements_method(
            calc, quantity_name, "read"
        ), f"{quantity_name} does not implement read()"


def test_selections_only_available_false_does_not_load_data(tmp_path, monkeypatch):
    # only_available=False must never attempt to load data
    calc = demo.calculation(tmp_path / "demo_calculation")

    def _fail(*_, **__):
        raise AssertionError("loadable_sources should not be called")

    monkeypatch.setattr(loadable, "loadable_sources", _fail)

    # neither the default nor the method-filtered call should load any data
    assert calc.selections()
    assert calc.selections(method="to_view")


def test_selections_collects_out_of_memory_errors(tmp_path, monkeypatch, capsys):
    # A quantity whose data is available but too large to load raises MemoryError.
    # selections() should not crash, should omit that quantity, and report the error.
    calc = demo.calculation(tmp_path / "demo_calculation")

    monkeypatch.setattr(loadable, "_schema_satisfied", lambda *_, **__: None)
    real_invoke = loadable._invoke

    def _invoke_with_oom(
        calculation,
        call_name,
        method_name,
        source_name,
        legacy_quantities,
        convention=None,
    ):
        if call_name == "density":
            raise MemoryError("Unable to allocate 5.00 GiB for array")
        return real_invoke(
            calculation,
            call_name,
            method_name,
            source_name,
            legacy_quantities,
            convention,
        )

    monkeypatch.setattr(loadable, "_invoke", _invoke_with_oom)

    result = calc.selections(only_available=True)

    # the oversized quantity is excluded from the loadable result
    assert "density" not in result
    # other quantities are still inspected and the call returns normally
    assert isinstance(result, dict)
    # the out-of-memory error is reported once at the end without crashing
    captured = capsys.readouterr()
    assert "density" in captured.out
    assert "out of memory" in captured.out.lower()
    assert "5.00 GiB" in captured.out


def test_selections_default_does_not_report_out_of_memory(
    tmp_path, monkeypatch, capsys
):
    # only_available=False never loads data, so it cannot trigger an OOM report
    calc = demo.calculation(tmp_path / "demo_calculation")

    def _fail(*_, **__):
        raise AssertionError("loadable_sources should not be called")

    monkeypatch.setattr(loadable, "loadable_sources", _fail)

    calc.selections()
    captured = capsys.readouterr()
    assert "out of memory" not in captured.out.lower()


def test_confirm_read_uses_public_call_name_for_fallback(tmp_path, monkeypatch):
    calc = demo.calculation(tmp_path / "demo_calculation")
    captured = {}

    monkeypatch.setattr(loadable, "_schema_satisfied", lambda *_, **__: None)

    def _record_invoke(
        calculation,
        call_name,
        method_name,
        source_name,
        legacy_quantities,
        convention=None,
    ):
        captured["call_name"] = call_name
        captured["method_name"] = method_name
        captured["source_name"] = source_name
        return True

    monkeypatch.setattr(loadable, "_invoke", _record_invoke)

    with contextlib.ExitStack() as stack:
        assert loadable._confirm_read(
            calc,
            "exciton.density",
            "exciton_density",
            DEFAULT_SELECTION,
            open_files={},
            stack=stack,
            cache={},
            legacy_quantities=set(),
        )

    assert captured["call_name"] == "exciton.density"
    assert captured["method_name"] == "read"
    assert captured["source_name"] == DEFAULT_SELECTION


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
