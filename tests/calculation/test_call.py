# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp import broadening, demo, exception
from py4vasp._calculation import _public_quantities, _quantity_object
from py4vasp._calculation._describe import call, describe
from py4vasp._calculation.electronic_minimization import ElectronicMinimization
from py4vasp._calculation.partial_density import PartialDensity


@pytest.fixture(scope="module")
def calculation(tmp_path_factory):
    return demo.calculation(tmp_path_factory.mktemp("call") / "data")


def labels(graph):
    return [series.label for series in graph.series]


def test_selection_projects_the_routine(calculation):
    actual = call(calculation, "dos", "to_graph", selection="Sr")
    assert labels(actual) == labels(calculation.dos.to_graph("Sr"))


def test_source_is_combined_with_the_selection(calculation):
    actual = call(
        calculation,
        "density",
        "to_contour",
        source="charge",
        selection="0",
        arguments={"a": 0.3},
    )
    expected = calculation.density.to_contour("charge(0)", a=0.3)
    assert labels(actual) == labels(expected)


def test_arguments_take_the_form_the_routine_expects(calculation):
    arguments = {"c": 0.25, "supercell": [2, 1], "normal": "x"}
    actual = call(calculation, "density", "to_contour", arguments=arguments)
    expected = calculation.density.to_contour(c=0.25, supercell=(2, 1), normal="x")
    assert np.array_equal(actual.series[0].supercell, expected.series[0].supercell)


def test_object_argument_becomes_its_dataclass(calculation):
    shape = {"name": "Lorentzian", "fwhm": 0.002}
    actual = call(calculation, "raman", "to_graph", arguments={"shape": shape})
    expected = calculation.raman.to_graph(shape=broadening.Lorentzian(fwhm=0.002))
    assert np.allclose(actual.series[0].y, expected.series[0].y)


def test_print_returns_the_text(calculation):
    assert call(calculation, "structure", "print") == str(calculation.structure)
    actual = call(calculation, "structure", "print", source="phonon")
    assert actual == calculation.structure.__str__("phonon")


@pytest.mark.parametrize(
    "steps, number_steps",
    ((3, 1), ("3", 1), ("-1", 1), ("0:5", 5), ("::2", 6), (":", 12)),
)
def test_steps_index_the_quantity(calculation, steps, number_steps):
    view = call(calculation, "structure", "to_view", steps=steps)
    assert len(view.positions) == number_steps


@pytest.mark.parametrize("steps", ("1:2:3:4", "a", "1.5", "2::0", "", True, 1.0))
def test_steps_must_be_an_index_or_a_slice(calculation, steps):
    with pytest.raises(exception.IncorrectUsage) as error:
        call(calculation, "structure", "to_view", steps=steps)
    assert error.value.field == "steps"


@pytest.mark.parametrize("steps", ("12", "-13"))
def test_step_beyond_the_source_names_the_steps(calculation, steps):
    with pytest.raises(exception.IncorrectUsage) as error:
        call(calculation, "structure", "to_view", steps=steps)
    assert error.value.field == "steps"


def test_single_step_beyond_the_history(calculation):
    # the default source of the DOS holds a single step, so step 5 does not exist
    with pytest.raises(exception.IncorrectUsage) as error:
        call(calculation, "dos", "to_graph", steps="5")
    assert error.value.field == "steps"


@pytest.mark.parametrize("quantity", ("structure", "energy"))
def test_range_of_steps_beyond_the_history(calculation, quantity):
    # the default sources hold 12 steps, so the range selects none of them
    with pytest.raises(exception.IncorrectUsage, match="20:30") as error:
        call(calculation, quantity, "read", steps="20:30")
    assert error.value.field == "steps"


def test_range_of_steps_that_reaches_past_the_history(calculation):
    # as in Python, a range is cut to the steps that exist
    actual = call(calculation, "energy", "read", steps="10:30")
    expected = calculation.energy[10:30].read()
    assert actual.keys() == expected.keys()


def test_steps_of_a_quantity_without_steps(calculation):
    with pytest.raises(exception.IncorrectUsage):
        call(calculation, "density", "to_contour", steps="0", arguments={"a": 0.1})


def test_default_steps_fit_a_single_structure(tmp_path):
    calculation = demo.calculation(tmp_path / "data", selection="surface")
    routine = describe(calculation, "structure", "print")
    assert routine["sources"][0]["steps"] == 1
    actual = call(calculation, "structure", "print", steps=routine["default_steps"])
    assert actual == call(calculation, "structure", "print")


def test_single_step_takes_no_range(calculation):
    with pytest.raises(exception.IncorrectUsage):
        call(calculation, "dos", "to_graph", steps="0:2")


@pytest.mark.parametrize(
    "arguments, field",
    (
        ({}, "cut"),  # none of the cut
        ({"a": 0.1, "b": 0.2}, "cut"),  # more than one of the cut
        ({"a": 0.1, "unknown": 1}, "unknown"),
        ({"a": 0.1, "normal": "w"}, "normal"),
        ({"a": 0.1, "selection": "0"}, "selection"),  # it has its own argument
        ({"a": "half"}, "a"),
    ),
)
def test_invalid_arguments_name_their_field(calculation, arguments, field):
    with pytest.raises(exception.IncorrectUsage) as error:
        call(calculation, "density", "to_contour", arguments=arguments)
    assert error.value.field == field


def test_required_argument(calculation):
    with pytest.raises(exception.IncorrectUsage, match="laser") as error:
        call(calculation, "raman", "intensity")
    assert error.value.field == "laser"
    actual = call(calculation, "raman", "intensity", arguments={"laser": 2.0})
    np.testing.assert_equal(actual, calculation.raman.intensity(laser=2.0))


def test_unknown_source(calculation):
    with pytest.raises(exception.IncorrectUsage, match="kpoints_opt") as error:
        call(calculation, "density", "to_contour", source="kpoints_opt")
    assert error.value.field == "source"


def test_routine_without_selection_takes_no_source(calculation):
    with pytest.raises(exception.IncorrectUsage) as error:
        call(calculation, "raman", "read", source="default")
    assert error.value.field == "source"


def test_routine_taking_only_a_selection(calculation):
    with pytest.raises(
        exception.IncorrectUsage, match="besides the selection"
    ) as error:
        call(calculation, "dos", "to_graph", arguments={"width": 1})
    assert error.value.field == "width"


def test_argument_without_annotation_is_passed_on(calculation):
    arguments = {"supercell": [2, 1, 1]}
    actual = call(calculation, "structure", "to_POSCAR", arguments=arguments)
    assert actual == calculation.structure.to_POSCAR(supercell=[2, 1, 1])
    assert actual != calculation.structure.to_POSCAR()


def test_data_read_comes_back_as_a_tree(calculation):
    actual = call(calculation, "band", "read")
    expected = calculation.band.read()
    assert actual.keys() == expected.keys()
    assert isinstance(actual["kpoint_labels"], np.ndarray)
    assert actual["kpoint_labels"].tolist() == expected["kpoint_labels"]
    assert type(actual["fermi_energy"]) is float
    assert np.array_equal(actual["bands"], expected["bands"])


def test_selection_of_projector_reads_as_text(calculation):
    # the two Sr atoms come first, written as a selection writes them
    actual = call(calculation, "projector", "read")
    assert actual["atom"]["Sr"] == "1:2"
    assert actual["atom"]["1"] == "1"


@pytest.mark.parametrize("selection", ("Xe", "Sr(", "Sr~"))
def test_selection_the_routine_cannot_read_names_the_selection(calculation, selection):
    with pytest.raises(exception.IncorrectUsage) as error:
        call(calculation, "dos", "to_graph", selection=selection)
    assert error.value.field == "selection"


_KINDS_OF_DATA = (None, "collinear", "noncollinear", "spin_texture", "surface")
# Values for the arguments a routine cannot run without.
_REQUIRED = {"a": 0.1, "laser": 2.0, "cutoff": 3.0}
# Routines that, on the demo data, stop for another reason before they read the selection.
_STOP_BEFORE_THE_SELECTION = {
    "bader_charge": "it needs the basins from bader_analysis",
    "band.to_quiver": "the demo bands lie on no grid of k points",
    "partial_density.plot": "without spin polarization, it ignores the selection",
    "partial_density.to_numpy": "without spin polarization, it ignores the selection",
    "partial_density.to_view": "without spin polarization, it ignores the selection",
}


@pytest.fixture(scope="module")
def kinds_of_data(tmp_path_factory):
    return [
        demo.calculation(tmp_path_factory.mktemp("kind") / "data", selection=kind)
        for kind in _KINDS_OF_DATA
    ]


def _routines_taking_a_selection():
    calculation = demo.calculation()
    for quantity, _ in _public_quantities():
        cls = type(_quantity_object(calculation, quantity))
        methods = [name for name in dir(cls) if not name.startswith(("_", "from"))]
        for method in [*methods, "__str__"]:
            if method == "is_available" or not callable(getattr(cls, method)):
                continue
            fields = describe(calculation, quantity, method)["fields"]
            if any(field["name"] == "selection" for field in fields):
                yield pytest.param(quantity, method, id=f"{quantity}.{method}")


@pytest.mark.parametrize("quantity, method", _routines_taking_a_selection())
def test_a_selection_the_routine_rejects_is_named(kinds_of_data, quantity, method):
    for routine in (method, f"{quantity}.{method}"):
        if reason := _STOP_BEFORE_THE_SELECTION.get(routine):
            pytest.skip(reason)
    calculation, arguments = _data_the_routine_runs_on(kinds_of_data, quantity, method)
    with pytest.raises(exception.Py4VaspError) as error:
        call(calculation, quantity, method, selection="zzz", arguments=arguments)
    if isinstance(error.value, exception.ModuleNotInstalled):
        pytest.skip(str(error.value))
    assert error.value.field == "selection", str(error.value)


@pytest.mark.parametrize(
    "quantity, method, selection",
    (
        ("dos", "to_graph", "default, Sr"),
        ("reaction_path", "read", "1~1"),
        ("reaction_path", "read", "Xe~O"),
        ("reaction_path", "read", "Sr~O"),
        ("reaction_path", "read", "1~99"),
        ("reaction_path", "to_path", None),
        ("phonon.mode", "to_view", "1:3"),
        ("phonon.mode", "to_view", "99"),
        ("bandgap", "to_graph", "fundamental, direct(up)"),
        ("dielectric_tensor", "to_INCAR", "clamped_ion, relaxed_ion"),
        ("potential", "to_quiver", "total(x)"),
        ("neighbor_list", "read", "Sr~Ti~O"),
    ),
)
def test_selection_a_routine_checks_itself_is_named(
    kinds_of_data, quantity, method, selection
):
    calculation, arguments = _data_the_routine_runs_on(kinds_of_data, quantity, method)
    with pytest.raises(exception.Py4VaspError) as error:
        call(calculation, quantity, method, selection=selection, arguments=arguments)
    if isinstance(error.value, exception.ModuleNotInstalled):
        pytest.skip(str(error.value))
    assert error.value.field == "selection", str(error.value)


def _data_the_routine_runs_on(kinds_of_data, quantity, method):
    "The first demo data the routine runs on, else the first that contains it."
    fallback = None
    for calculation in kinds_of_data:
        routine = describe(calculation, quantity, method)
        if not routine["available"]:
            continue
        arguments = _required_arguments(routine)
        fallback = fallback or (calculation, arguments)
        try:
            call(calculation, quantity, method, arguments=arguments)
        except exception.ModuleNotInstalled as error:
            pytest.skip(str(error))  # the core installation lacks an optional package
        except exception.Py4VaspError:
            continue
        return calculation, arguments
    if fallback is None:
        pytest.skip(f"the demo data contains no {quantity}")
    return fallback


def _required_arguments(routine):
    names = [field["name"] for field in routine["fields"] if field["required"]]
    names += [
        group["fields"][0] for group in routine["groups"].values() if group["required"]
    ]
    names = [name for name in names if name != "selection"]
    if missing := [name for name in names if name not in _REQUIRED]:
        pytest.skip(f"the test has no value for {missing}")
    return {name: _REQUIRED[name] for name in names}


def test_selection_of_the_electronic_minimization_is_named(raw_data):
    quantity = ElectronicMinimization.from_data(raw_data.electronic_minimization())
    with pytest.raises(exception.Py4VaspError) as error:
        quantity.read(selection="zzz")
    assert error.value.field == "selection"


def test_spin_of_the_partial_density_is_named(raw_data):
    raw = raw_data.partial_density("Sr2TiO4 spin_polarized")
    with pytest.raises(exception.IncorrectUsage) as error:
        PartialDensity.from_data(raw).to_numpy("zzz")
    assert error.value.field == "selection"


@pytest.mark.parametrize("key", ("O(", "Xe", "1:99"))
def test_masses_are_named_although_their_keys_look_like_selections(calculation, key):
    with pytest.raises(exception.IncorrectUsage) as error:
        call(calculation, "phonon.mode", "to_view", arguments={"masses": {key: 16.0}})
    assert error.value.field == "masses"


def test_missing_data_names_no_field(calculation):
    with pytest.raises(exception.NoData) as error:
        call(calculation, "structure", "to_view", source="final")
    assert error.value.field is None


def test_other_errors_of_the_routine_name_no_field(calculation):
    # only the routine knows that a supercell must repeat the cell at least once
    with pytest.raises(exception.IncorrectUsage) as error:
        call(calculation, "structure", "to_view", arguments={"supercell": [0, 1, 1]})
    assert error.value.field is None
