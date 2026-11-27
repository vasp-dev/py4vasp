# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""A tool describes every routine it offers with backend.inspect.

A routine that cannot be described would fail when a user clicks it, so every routine
of every quantity is described here, for a calculation with data and for one without.
"""

import inspect
import json
import typing

import pytest
from vasp import backend

import py4vasp
from py4vasp import demo, exception
from py4vasp._calculation import _public_quantities, _quantity_object

_ROUTINES = (
    "read",
    "to_dict",
    "plot",
    "to_graph",
    "to_view",
    "to_contour",
    "to_quiver",
    "to_stm",
    "print",
    "__str__",
    "selections",
    "number_steps",
)


@pytest.fixture(scope="module")
def calculations(tmp_path_factory):
    with_data = demo.calculation(tmp_path_factory.mktemp("inspect") / "data")
    without_data = py4vasp.Calculation.from_path(tmp_path_factory.mktemp("empty"))
    return {"with data": with_data, "without data": without_data}


def _cases():
    "Every public routine of every quantity, not only those a tool offers today."
    calculation = demo.calculation()
    for quantity, _ in _public_quantities():
        instance = _quantity_object(calculation, quantity)
        for method in sorted(set(_ROUTINES) | set(_public_routines(instance))):
            routine = getattr(instance, method, None)
            if routine is not None and routine is not getattr(object, method, None):
                yield pytest.param(quantity, method, id=f"{quantity}.{method}")


def _public_routines(instance):
    for name in dir(type(instance)):
        member = inspect.getattr_static(type(instance), name)
        if not name.startswith("_") and inspect.isfunction(member):
            yield name


@pytest.mark.parametrize("data", ("with data", "without data"))
@pytest.mark.parametrize("quantity, method", _cases())
def test_every_routine_is_described(calculations, data, quantity, method):
    routine = backend.inspect(calculations[data], quantity, method)
    assert json.loads(json.dumps(routine)) == routine
    instance = _quantity_object(calculations[data], quantity)
    called = {"plot": _plotted_by(instance)}.get(method, method)
    parameters = inspect.signature(getattr(instance, called)).parameters.values()
    expected = [
        p.name for p in parameters if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    assert [field["name"] for field in routine["fields"]] == expected
    if data == "without data":
        assert routine["available"] is False


def _plotted_by(instance):
    return "to_graph" if hasattr(instance, "to_graph") else "to_view"


def test_keys_follow_the_typed_dictionaries(calculations):
    routine = backend.inspect(calculations["with data"], "density", "to_contour")
    assert set(routine) == set(typing.get_type_hints(backend.Routine))
    assert set(routine["sources"][0]) == set(typing.get_type_hints(backend.Source))
    assert set(routine["groups"]["cut"]) == set(typing.get_type_hints(backend.Group))
    returned = backend.inspect(calculations["with data"], "phonon.band", "read")
    assert set(returned["returns"][0]) == set(typing.get_type_hints(backend.Return))
    field_keys = set(typing.get_type_hints(backend.Field))
    for field in routine["fields"]:
        assert set(field) <= field_keys
        assert backend.Field.__required_keys__ <= set(field)


def test_unknown_routine_is_an_incorrect_usage(calculations):
    with pytest.raises(exception.IncorrectUsage):
        backend.inspect(calculations["with data"], "dos", "to_view")


def test_call_takes_what_inspect_describes(calculations):
    calculation = calculations["with data"]
    routine = backend.inspect(calculation, "density", "to_contour")
    cut = next(field for field in routine["fields"] if field["name"] == "a")
    arguments = {"a": cut["type"]["maximum"] / 2, "supercell": [2, 1]}
    graph = backend.call(calculation, "density", "to_contour", arguments=arguments)
    assert graph.series[0].supercell.tolist() == [2, 1]


def test_call_rejects_what_inspect_does_not_describe(calculations):
    with pytest.raises(exception.IncorrectUsage):
        backend.call(calculations["with data"], "dos", "to_graph", arguments={"x": 1})
