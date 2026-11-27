# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Checks every routine a quantity offers to tools that present it to a user.

A tool builds a form for each of these routines from what py4vasp declares: the
docstring describes the routine and its parameters, the annotations their types. So
the docstrings must parse and read as prose outside the examples.
"""

import inspect
import re
import typing

import pytest

from py4vasp._calculation import _ensure_all_quantities_imported
from py4vasp._calculation.dispatch import _REGISTRY
from py4vasp._util import docstring, slicing

ROUTINES = (
    "read",
    "to_dict",
    "plot",
    "to_graph",
    "to_view",
    "to_contour",
    "to_quiver",
    "to_stm",
    "_to_database",
    "print",
    "__str__",
    "selections",
    "number_steps",
)


def _quantity_classes():
    _ensure_all_quantities_imported()
    for name, entry in _REGISTRY.items():
        if name.startswith("_"):
            continue
        if isinstance(entry, dict):
            yield from (cls for key, cls in entry.items() if not key.startswith("_"))
        else:
            yield entry


def _routines():
    for cls in _quantity_classes():
        for name in ROUTINES:
            routine = getattr(cls, name, None)
            if routine is None or routine is getattr(object, name, None):
                continue
            yield pytest.param(routine, id=f"{cls.__name__}.{name}")


@pytest.mark.parametrize("routine", _routines())
def test_docstring_is_prose_outside_the_examples(routine):
    parsed = docstring.parse(inspect.getdoc(routine))
    prose = [parsed.summary, parsed.extended_summary]
    prose += [entry.description for entry in parsed.parameters + parsed.returns]
    prose += [text for name, text in parsed.sections.items() if name != "Examples"]
    assert not any(">>>" in text for text in prose)


# A tool shows the documentation of a routine without its Examples section, so the
# rest must not send the reader there; it can give a short example inline instead.
_POINTS_TO_THE_EXAMPLES = re.compile(
    r"\bexamples?\s+(below|above)\b|\bthe\s+examples\b|\bexamples\s+section\b",
    re.IGNORECASE,
)


def _all_public_routines():
    for cls in _quantity_classes():
        for name in sorted(name for name in dir(cls) if not name.startswith("_")):
            routine = getattr(cls, name)
            if callable(routine) and inspect.getdoc(routine):
                yield pytest.param(routine, id=f"{cls.__name__}.{name}")


@pytest.mark.parametrize("routine", _all_public_routines())
def test_prose_does_not_point_to_the_examples(routine):
    parsed = docstring.parse(inspect.getdoc(routine))
    prose = [parsed.summary, parsed.extended_summary]
    prose += [entry.description for entry in parsed.parameters + parsed.returns]
    prose += [text for name, text in parsed.sections.items() if name != "Examples"]
    pointers = [text for text in prose if _POINTS_TO_THE_EXAMPLES.search(text)]
    assert pointers == []


@pytest.mark.parametrize("routine", _routines())
def test_every_parameter_and_the_return_are_annotated(routine):
    hints = typing.get_type_hints(routine, include_extras=True)
    parameters = inspect.signature(routine).parameters
    assert [name for name in parameters if name not in hints and name != "self"] == []
    assert "return" in hints


@pytest.mark.parametrize("routine", _routines())
def test_documented_types_match_the_annotations(routine):
    hints = typing.get_type_hints(routine, include_extras=True)
    for entry in docstring.parse(inspect.getdoc(routine)).parameters:
        assert entry.name in hints, f"{entry.name} is documented but not annotated"
        assert entry.type == docstring.format_type(hints[entry.name]), entry.name


@pytest.mark.parametrize("routine", _routines())
def test_cut_plane_takes_exactly_one_of_a_b_c(routine):
    hints = typing.get_type_hints(routine, include_extras=True)
    cuts = [name for name in "abc" if name in inspect.signature(routine).parameters]
    assert cuts in ([], ["a", "b", "c"])
    assert all(hints[name] == slicing.CutFraction for name in cuts)
