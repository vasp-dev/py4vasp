# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import json
import typing

import numpy as np
import pytest

from py4vasp import demo, exception
from py4vasp._calculation import _public_quantities, _quantity_object
from py4vasp._calculation._describe import (
    _default_steps,
    _hints,
    _parse_steps,
    _step_selection,
    describe,
)
from py4vasp._calculation.current_density import CurrentDensity
from py4vasp._calculation.electron_phonon_bandgap import ElectronPhononBandgap
from py4vasp._calculation.electron_phonon_self_energy import ElectronPhononSelfEnergy
from py4vasp._calculation.electron_phonon_transport import ElectronPhononTransport
from py4vasp._calculation.electronic_minimization import ElectronicMinimization
from py4vasp._util import form
from py4vasp.raw import VaspData


@pytest.fixture(scope="module")
def calculation(tmp_path_factory):
    return demo.calculation(tmp_path_factory.mktemp("describe") / "data")


def _field(routine, name):
    return next(field for field in routine["fields"] if field["name"] == name)


def test_describes_the_routine_from_its_docstring(calculation):
    routine = describe(calculation, "dos", "to_graph")
    assert routine["quantity"] == "dos"
    assert routine["method"] == "to_graph"
    assert routine["summary"] == "Read the DOS and convert it into a graph."
    assert routine["description"].startswith("The x axis is the energy mesh")
    assert ">>>" not in routine["description"]
    assert routine["available"] is True
    assert json.loads(json.dumps(routine)) == routine


def test_sources_with_aliases_availability_and_steps(calculation):
    sources = describe(calculation, "density", "to_contour")["sources"]
    default = {"name": "default", "available": True, "steps": 1}
    assert sources[0] == {
        **default,
        "aliases": ["charge", "n", "charge_density", "electronic_charge_density"],
    }
    tau = next(source for source in sources if source["name"] == "tau")
    assert tau["aliases"] == ["kinetic_energy", "kinetic_energy_density"]


def test_steps_of_a_trajectory_and_of_a_single_structure(calculation):
    routine = describe(calculation, "structure", "to_view")
    assert routine["steps"] == "slice"
    steps = {source["name"]: source["steps"] for source in routine["sources"]}
    assert steps["default"] == 12
    assert steps["phonon"] == 1
    assert describe(calculation, "dos", "to_graph")["steps"] == "single"
    energy = describe(calculation, "energy", "to_graph")
    assert energy["sources"][0]["steps"] == len(calculation.energy[:].to_numpy())
    assert describe(calculation, "density", "to_contour")["steps"] is None


def test_selection_lists_the_axes_of_the_routine(calculation):
    selection = _field(describe(calculation, "dos", "to_graph"), "selection")
    assert selection["nullable"] is True
    assert selection["type"]["kind"] == "selection"
    assert selection["type"]["operators"] == form.OPERATORS
    names = [axis["name"] for axis in selection["type"]["axes"]]
    assert names == ["atom", "orbital", "spin"]
    assert selection["description"].startswith("A string specifying the projection")


def test_cut_plane_forms_a_required_exclusive_group(calculation):
    routine = describe(calculation, "density", "to_contour")
    a = _field(routine, "a")
    assert a["type"] == {"kind": "number", "minimum": 0.0, "maximum": 1.0}
    assert a["group"] == "cut"
    normal = _field(routine, "normal")
    assert normal["type"]["choices"] == ["x", "y", "z", "auto"]
    group = {"exclusive": True, "required": True, "fields": ["a", "b", "c"]}
    assert routine["groups"] == {"cut": group}


def test_aliases_are_documented_where_they_point(calculation):
    to_dict = describe(calculation, "dos", "to_dict")
    read = describe(calculation, "dos", "read")
    assert to_dict["summary"] == read["summary"]
    assert _field(to_dict, "selection")["description"] != ""
    printed = describe(calculation, "structure", "__str__")
    assert printed["summary"] == "Print a string representation of this quantity."


def test_plot_is_described_as_the_routine_it_calls(calculation):
    plot = describe(calculation, "dos", "plot")
    to_graph = describe(calculation, "dos", "to_graph")
    assert plot["fields"] == to_graph["fields"]
    view = describe(calculation, "structure", "plot")
    assert [field["name"] for field in view["fields"]] == [
        "supercell",
        "ion_types",
        "selection",
    ]


def test_required_argument(calculation):
    cutoff = _field(describe(calculation, "neighbor_list", "read"), "cutoff")
    assert cutoff["required"] is True
    assert "default" not in cutoff


def test_group_member(calculation):
    routine = describe(calculation, "phonon.band", "to_graph")
    assert routine["quantity"] == "phonon.band"
    assert routine["available"] is True


def test_missing_data_is_reported_not_raised(calculation):
    routine = describe(calculation, "local_moment", "to_view")
    assert routine["available"] is False
    assert not any(source["available"] for source in routine["sources"])
    assert _field(routine, "selection")["type"]["axes"] == []


@pytest.mark.parametrize("quantity, method", (("no_such", "read"), ("dos", "no_such")))
def test_unknown_quantity_or_method(calculation, quantity, method):
    with pytest.raises(exception.IncorrectUsage):
        describe(calculation, quantity, method)


@pytest.fixture(scope="module")
def kinds_of_data(tmp_path_factory):
    kinds = (None, "collinear", "noncollinear", "metal", "surface", "perovskite")
    return [
        demo.calculation(tmp_path_factory.mktemp("kind") / "data", selection=kind)
        for kind in kinds
    ]


def _quantities_with_steps():
    for quantity, _ in _public_quantities():
        cls = type(_quantity_object(demo.calculation(), quantity))
        if _step_selection(cls) is not None:
            yield quantity


@pytest.mark.parametrize("quantity", _quantities_with_steps())
def test_every_available_source_counts_its_steps(kinds_of_data, quantity):
    counted = 0
    for calculation in kinds_of_data:
        for source in describe(calculation, quantity, "read")["sources"]:
            if source["available"]:
                assert isinstance(source["steps"], int) and source["steps"] >= 1
                counted += 1
    if not counted:
        pytest.skip(f"the demo data contains no {quantity}")


def test_steps_follow_the_structure(calculation):
    structure = calculation.structure[:].number_steps()
    for quantity in ("neighbor_list", "reaction_path"):
        sources = describe(calculation, quantity, "read")["sources"]
        assert sources[0]["steps"] == structure


def test_steps_of_the_electronic_minimization(raw_data):
    raw = raw_data.electronic_minimization()
    quantity = ElectronicMinimization.from_data(raw)
    assert quantity._number_steps("default") == len(raw.is_elmin_converged)


@pytest.mark.parametrize(
    "cls, make_raw",
    (
        (ElectronPhononTransport, lambda raw: raw.electron_phonon_transport("default")),
        (
            ElectronPhononSelfEnergy,
            lambda raw: raw.electron_phonon_self_energy("default"),
        ),
        (ElectronPhononBandgap, lambda raw: raw.electron_phonon_band_gap("default")),
        (CurrentDensity, lambda raw: raw.current_density("all")),
    ),
)
def test_quantity_without_instances_is_not_available(raw_data, cls, make_raw):
    # a run may write the group of a quantity without computing any instance of it
    raw = make_raw(raw_data)
    assert cls.from_data(raw).is_available() is True
    empty = dataclasses.replace(raw, valid_indices=[])
    assert cls.from_data(empty).is_available() is False


@pytest.mark.parametrize(
    "cls, make_raw",
    (
        (
            ElectronPhononSelfEnergy,
            lambda raw: raw.electron_phonon_self_energy("default"),
        ),
        (ElectronPhononBandgap, lambda raw: raw.electron_phonon_band_gap("default")),
    ),
)
def test_without_scattering_approximation_only_reading_works(raw_data, cls, make_raw):
    # VASP does not write the scattering approximation in every run; the instances
    # can still be read, but printing and listing them needs it
    raw = make_raw(raw_data)
    missing = [VaspData(None) for _ in raw.valid_indices]
    raw = dataclasses.replace(raw, scattering_approximation=missing)
    quantity = cls.from_data(raw)
    assert quantity.is_available() is True
    assert quantity.is_available(method="read") is True
    assert quantity.is_available(method="to_dict") is True
    assert quantity.is_available(method="print") is False
    assert quantity.is_available(method="selections") is False


@pytest.mark.parametrize(
    "quantity, method",
    (("dos", "to_frame"), ("energy", "to_numpy"), ("structure", "to_ase")),
)
def test_parameter_without_annotation_takes_anything(calculation, quantity, method):
    routine = describe(calculation, quantity, method)
    unannotated = [f for f in routine["fields"] if f["name"] != "selection"]
    assert all(field["type"] == {"kind": "any"} for field in unannotated)


def test_annotations_that_cannot_be_resolved_are_taken_as_anything():
    def routine(first: "NotDefinedAnywhere", second: int) -> "pd.DataFrame":
        pass

    assert _hints(routine) == {"first": typing.Any, "second": int}


def test_neighbor_list_takes_a_single_step(calculation):
    # the neighbors are found in one structure, a range of steps is not supported
    assert describe(calculation, "neighbor_list", "read")["steps"] == "single"


def test_returns_and_notes_are_described_apart(calculation):
    routine = describe(calculation, "phonon.band", "read")
    (returned,) = routine["returns"]
    assert returned["name"] == ""
    assert returned["type"] == "dict"
    assert returned["description"].startswith("Contains the **q**-point path")
    assert routine["notes"].startswith("VASP reports the branches in THz")
    assert "Contains the" not in routine["description"]
    assert "VASP reports" not in routine["description"]


def test_return_without_type(calculation):
    # py4vasp writes "-" where a docstring gives no type for what a routine returns
    (returned,) = describe(calculation, "energy", "selections")["returns"]
    assert returned["type"] == ""
    assert returned["description"] != ""


def test_default_steps(calculation):
    # written as the steps of call: the last step, or all of them
    assert describe(calculation, "structure", "to_view")["default_steps"] == "-1"
    assert describe(calculation, "dos", "to_graph")["default_steps"] == "-1"
    assert describe(calculation, "reaction_path", "read")["default_steps"] == ":"
    assert describe(calculation, "density", "to_contour")["default_steps"] is None


_READ_ARGUMENTS = {
    "neighbor_list": {"cutoff": 3.0},
    "reaction_path": {"selection": "1~2"},
}


@pytest.mark.parametrize("quantity", _quantities_with_steps())
def test_default_steps_are_what_reading_without_steps_reads(kinds_of_data, quantity):
    # With a single step, every default reads the same; only more steps tell them apart.
    compared = 0
    for calculation in kinds_of_data:
        routine = describe(calculation, quantity, "read")
        source = routine["sources"][0]
        if not source["available"] or (source["steps"] or 0) < 2:
            continue
        instance = _quantity_object(calculation, quantity)
        arguments = _READ_ARGUMENTS.get(quantity, {})
        try:
            expected = instance.read(**arguments)
        except exception.IncorrectUsage:
            continue  # e.g. a structure of one atom has no pair to measure
        indexed = instance[_parse_steps(routine["default_steps"])]
        np.testing.assert_equal(indexed.read(**arguments), expected)
        compared += 1
    if not compared:
        pytest.skip(f"the demo data holds no {quantity} of more than one step")


def test_default_steps_of_the_electronic_minimization(raw_data):
    first, last = (raw_data.electronic_minimization() for _ in range(2))
    joined = np.concatenate((first.convergence_data, last.convergence_data))
    raw = dataclasses.replace(
        first, convergence_data=VaspData(joined), is_elmin_converged=[0, 0]
    )
    quantity = ElectronicMinimization.from_data(raw)
    kind = _step_selection(ElectronicMinimization)
    default = _parse_steps(_default_steps(ElectronicMinimization, kind))
    np.testing.assert_equal(quantity[default].read(), quantity.read())
