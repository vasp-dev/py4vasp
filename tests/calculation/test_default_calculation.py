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
    calc = Calculation.from_path(tmp_path)
    assert calc.selections() == {}


def test_selections_on_demo_calculation(tmp_path):
    calc = demo.calculation(tmp_path / "demo_calculation")
    actual = calc.selections()
    # quantities with data expose ready-to-evaluate snippets per loadable selection
    expected = {
        "band": {
            "default": "calculation.band.read()",
            "kpoints_opt": "calculation.band.read(selection='kpoints_opt')",
        },
        "dos": {
            "default": "calculation.dos.read()",
            "kpoints_opt": "calculation.dos.read(selection='kpoints_opt')",
        },
        "energy": {"default": "calculation.energy.read()"},
        "exciton.density": {"default": "calculation.exciton.density.read()"},
        "force": {"default": "calculation.force.read()"},
        "nics": {"default": "calculation.nics.read()"},
        "partial_density": {"default": "calculation.partial_density.read()"},
        "potential": {"default": "calculation.potential.read()"},
        "stress": {"default": "calculation.stress.read()"},
        "system": {"default": "calculation.system.read()"},
        "velocity": {"default": "calculation.velocity.read()"},
    }
    for quantity, snippets in expected.items():
        assert actual[quantity] == snippets
    # read is decided from the files: the kinetic part (tau) is present as a dataset
    assert actual["density"] == {
        "default": "calculation.density.read()",
        "tau": "calculation.density['tau'].read()",
    }
    # structure is not migrated yet, so only its default source can be addressed
    assert actual["structure"] == {"default": "calculation.structure.read()"}
    # the result is sorted by quantity name
    assert list(actual) == sorted(actual)


def test_selections_excludes_selections_that_do_not_load(tmp_path):
    calc = demo.calculation(tmp_path / "demo_calculation")
    actual = calc.selections()
    # current_density cannot be read without specifying a cut plane -> excluded
    assert "current_density" not in actual
    # a non-default source of a not-yet-migrated quantity cannot be addressed
    assert set(actual["structure"]) == {"default"}


def test_selections_snippets_are_evaluable(tmp_path):
    calculation = demo.calculation(tmp_path / "demo_calculation")
    # to_view confirms loadability by invoking the method, so every snippet must run
    for snippets in calculation.selections(method="to_view").values():
        for snippet in snippets.values():
            eval(snippet)  # the generated snippet must run without error


def test_selections_excludes_quantities_without_data(tmp_path):
    calc = demo.calculation(tmp_path / "demo_calculation")
    actual = calc.selections()
    absent = (
        "bandgap",
        "born_effective_charge",
        "dielectric_function",
        "dielectric_tensor",
        "elastic_modulus",
        "internal_strain",
        "piezoelectric_tensor",
        "polarization",
    )
    for quantity in absent:
        assert quantity not in actual


def test_selections_filtered_by_method(tmp_path):
    calc = demo.calculation(tmp_path / "demo_calculation")
    viewable = calc.selections(method="to_view")
    full = calc.selections()
    # only quantities implementing the method (and loadable via it) are reported
    assert set(viewable) <= set(full)
    assert viewable.keys() >= {"density", "potential", "structure"}
    # quantities without a to_view method are excluded
    for quantity in ("band", "dos", "energy", "stress"):
        assert quantity not in viewable
    # the snippets call the requested method on the matching selection
    assert viewable["density"] == {"default": "calculation.density.to_view()"}
    assert viewable["structure"] == {"default": "calculation.structure.to_view()"}
    # the selections per quantity are still consistent with the unfiltered result
    for quantity, snippets in viewable.items():
        assert set(snippets) <= set(full[quantity])


def test_selections_with_method_on_empty_path(tmp_path):
    calc = Calculation.from_path(tmp_path)
    assert calc.selections(method="to_view") == {}


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
