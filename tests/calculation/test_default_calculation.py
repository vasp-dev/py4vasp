# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import contextlib

import pytest

from py4vasp import Calculation, calculation, demo
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
    absent_in_loadable = {
        "bandgap",
        "born_effective_charge",
        "dielectric_function",
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
