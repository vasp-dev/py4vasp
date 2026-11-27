# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp import broadening, demo, exception
from py4vasp._calculation._describe import call, describe


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


def test_other_errors_of_the_routine_name_no_field(calculation):
    # only the routine knows that a supercell must repeat the cell at least once
    with pytest.raises(exception.IncorrectUsage) as error:
        call(calculation, "structure", "to_view", arguments={"supercell": [0, 1, 1]})
    assert error.value.field is None
