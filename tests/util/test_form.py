# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import collections.abc
import dataclasses
import inspect
import json
import types
import typing

import numpy as np
import numpy.typing
import pytest

from py4vasp import exception
from py4vasp._util import form, slicing


@dataclasses.dataclass
class Settings:
    """Settings of a computation.

    Parameters
    ----------
    width : float
        The width.
    """

    width: float = 4.0
    count: int | None = None


class Opaque:
    pass


@pytest.mark.parametrize(
    "annotation, expected",
    (
        (bool, {"kind": "boolean"}),
        (int, {"kind": "integer"}),
        (float, {"kind": "number"}),
        (str, {"kind": "text"}),
        (typing.Any, {"kind": "any"}),
        (types.EllipsisType, {"kind": "ellipsis"}),
        (
            typing.Literal["x", "y", "auto"],
            {"kind": "single_choice", "choices": ["x", "y", "auto"]},
        ),
        (
            tuple[int, int, int],
            {"kind": "list", "items": {"kind": "integer"}, "length": 3},
        ),
        (
            collections.abc.Sequence[str],
            {"kind": "list", "items": {"kind": "text"}, "length": None},
        ),
        (list[float], {"kind": "list", "items": {"kind": "number"}, "length": None}),
        (
            typing.Mapping[str, float],
            {"kind": "mapping", "keys": {"kind": "text"}, "values": {"kind": "number"}},
        ),
        (
            int | tuple[int, int],
            {
                "kind": "union",
                "options": [
                    {"kind": "integer"},
                    {"kind": "list", "items": {"kind": "integer"}, "length": 2},
                ],
            },
        ),
        (Opaque, {"kind": "unsupported", "name": "Opaque"}),
    ),
)
def test_type_spec(annotation, expected):
    assert form.type_spec(annotation) == expected


def test_array_like_is_a_number_or_a_list_of_numbers():
    numbers = {"kind": "list", "items": {"kind": "number"}, "length": None}
    expected = {"kind": "union", "options": [{"kind": "number"}, numbers]}
    assert form.type_spec(numpy.typing.ArrayLike) == expected
    assert form.type_spec(numpy.typing.ArrayLike | None) == expected


def test_type_spec_leaves_out_none():
    assert form.type_spec(float | None) == {"kind": "number"}
    assert form.type_spec(typing.Optional[str]) == {"kind": "text"}
    expected = {"kind": "union", "options": [{"kind": "ellipsis"}, {"kind": "number"}]}
    assert form.type_spec(types.EllipsisType | float | None) == expected


def test_range_bounds_a_number():
    annotation = typing.Annotated[float | None, slicing.Range(minimum=0.0)]
    assert form.type_spec(annotation) == {"kind": "number", "minimum": 0.0}
    expected = {"kind": "number", "minimum": 0.0, "maximum": 1.0}
    assert form.type_spec(slicing.CutFraction) == expected


@pytest.mark.parametrize(
    "annotation, expected",
    (
        (float, False),
        (float | None, True),
        (typing.Optional[int], True),
        (slicing.CutFraction, True),
        (typing.Any, True),
    ),
)
def test_is_nullable(annotation, expected):
    assert form.is_nullable(annotation) == expected


def test_find_marker():
    group = form.find_marker(slicing.CutFraction, slicing.ExclusiveGroup)
    assert group == slicing.ExclusiveGroup("cut", required=True)
    assert form.find_marker(float | None, slicing.ExclusiveGroup) is None


def test_dataclass_is_an_object_with_fields():
    expected = {
        "kind": "object",
        "name": "Settings",
        "fields": [
            form.field("width", float, 4.0, "The width."),
            form.field("count", int | None, None),
        ],
    }
    assert form.type_spec(Settings) == expected


def test_field():
    actual = form.field("a", slicing.CutFraction, None, "The cut.")
    assert actual == {
        "name": "a",
        "description": "The cut.",
        "required": False,
        "nullable": True,
        "default": None,
        "type": {"kind": "number", "minimum": 0.0, "maximum": 1.0},
        "group": "cut",
    }


def test_required_field_has_no_default():
    actual = form.field("cutoff", float, inspect.Parameter.empty)
    assert actual == {
        "name": "cutoff",
        "description": "",
        "required": True,
        "nullable": False,
        "type": {"kind": "number"},
    }


@pytest.mark.parametrize(
    "value, expected",
    (
        (None, None),
        (True, True),
        (2, 2),
        (0.5, 0.5),
        ("total", "total"),
        ((1, 2), [1, 2]),
        (np.float64(1.5), 1.5),
        (np.array([1, 2]), [1, 2]),
        (..., "..."),
        (Settings(), {"width": 4.0, "count": None}),
    ),
)
def test_to_json(value, expected):
    actual = form.to_json(value)
    assert actual == expected
    assert json.loads(json.dumps(actual)) == expected


def test_operators_follow_the_selection_parser():
    assert form.OPERATORS == {
        "separators": [",", " "],
        "nesting": ["(", ")"],
        "range": ":",
        "pair": "~",
        "arithmetic": ["+", "-"],
        "assignment": "=",
    }


def test_atom_axis_lists_every_element_once():
    actual = form.atom_axis(["Sr", "Sr", "Ti", "O", "O", "O", "O"])
    expected = {"kind": "atom", "name": "atom", "elements": ["Sr", "Ti", "O"]}
    assert actual == {**expected, "indices": [1, 7]}


def test_choice_axis():
    actual = form.choice_axis(
        "component",
        ["total", form.token("3", aliases=["sigma_z", "m"], description="along z")],
    )
    assert actual == {
        "kind": "choice",
        "name": "component",
        "tokens": [
            {"name": "total", "aliases": [], "description": ""},
            {"name": "3", "aliases": ["sigma_z", "m"], "description": "along z"},
        ],
    }


def test_pair_axis():
    members = form.choice_axis("element", ["Sr", "Ti"])
    actual = form.pair_axis("pair", members)
    assert actual == {
        "kind": "pair",
        "name": "pair",
        "members": members,
        "distinct": True,
    }
    same_allowed = form.pair_axis("pair", members, distinct=False)
    assert same_allowed["distinct"] is False


def test_pairs_of_the_same_member_are_listed_unless_distinct():
    members = form.choice_axis("element", ["Sr", "O"])
    axes = [form.pair_axis("pair", members, distinct=False)]
    expected = ["Sr~Sr", "Sr~O", "O~Sr", "O~O"]
    assert form.selections_from_axes(axes) == {"pair": expected}


def test_index_axis():
    assert form.index_axis("mode", 21) == {
        "kind": "index",
        "name": "mode",
        "indices": [1, 21],
    }


def test_assignment_axis_describes_the_values_of_each_key():
    keys = {"band": form.index_axis("band", 8), "T": form.choice_axis("T", ["300.0"])}
    actual = form.assignment_axis("instance", keys)
    assert actual == {"kind": "assignment", "name": "instance", "keys": keys}


def test_axes_from_selections_skips_the_sources():
    selections = {
        "dos": ["default", "kpoints_opt"],
        "atom": ["Sr", "Ti", "O", "1", "2", "3", "4"],
        "spin": ["total"],
    }
    actual = form.axes_from_selections(selections, ("atom", "spin"))
    expected_atom = {"kind": "atom", "name": "atom", "elements": ["Sr", "Ti", "O"]}
    assert actual == [
        {**expected_atom, "indices": [1, 4]},
        form.choice_axis("spin", ["total"]),
    ]


@dataclasses.dataclass(kw_only=True)
class Wide:
    fwhm: float | None = None


@dataclasses.dataclass(kw_only=True)
class Narrow:
    fwhm: float | None = None


@pytest.mark.parametrize(
    "annotation, value, expected",
    (
        (bool, True, True),
        (int, 3, 3),
        (int, 3.0, 3),
        (float, 2, 2.0),
        (str, "total", "total"),
        (typing.Any, {"isolevel": 0.1}, {"isolevel": 0.1}),
        (float | None, None, None),
        (typing.Literal["x", "auto"] | None, "auto", "auto"),
        (types.EllipsisType | collections.abc.Sequence[float] | None, "...", ...),
        (
            types.EllipsisType | collections.abc.Sequence[float] | None,
            [1, 2],
            [1.0, 2.0],
        ),
        (int | tuple[int, int] | None, 2, 2),
        (int | tuple[int, int] | None, [2, 3], (2, 3)),
        (collections.abc.Sequence[str] | None, ["Sr", "Ti"], ["Sr", "Ti"]),
        (typing.Mapping[str, float], {"Sr": 87.6}, {"Sr": 87.6}),
        (slicing.CutFraction, 1.3, 1.3),  # the range describes, it does not reject
        (numpy.typing.ArrayLike, [0.1, 0.2], [0.1, 0.2]),
        (Settings, {"width": 2}, Settings(width=2.0)),
        (Wide | Narrow | None, {"name": "Narrow", "fwhm": 1}, Narrow(fwhm=1.0)),
    ),
)
def test_from_json(annotation, value, expected):
    actual = form.from_json(value, annotation)
    assert actual == expected
    assert type(actual) is type(expected)


@pytest.mark.parametrize(
    "annotation, value",
    (
        (bool, 1),
        (int, True),
        (int, 2.5),
        (float, "1.0"),
        (str, 3),
        (float, None),
        (typing.Literal["x", "auto"], "w"),
        (tuple[int, int], [1, 2, 3]),
        (collections.abc.Sequence[str], "Sr"),
        (int | tuple[int, int], "2"),
        (Settings, {"width": 1, "unknown": 2}),
        (Settings, [1]),
        (Wide | Narrow, {"fwhm": 1}),  # which of the two is meant?
        (Wide | Narrow, {"name": "Broad", "fwhm": 1}),
        (Opaque, {}),
    ),
)
def test_from_json_rejects_what_the_annotation_does_not_allow(annotation, value):
    with pytest.raises(exception.IncorrectUsage):
        form.from_json(value, annotation)


def test_selections_from_axes_lists_every_token_in_the_old_format():
    axes = [
        form.atom_axis(["Sr", "Ti", "O"]),
        form.choice_axis("gap", ["fundamental", form.token("direct", ["d"])]),
        form.pair_axis("spin_components", form.choice_axis("axis", ["x", "y"])),
        form.index_axis("mode", 3),
        form.assignment_axis("filter", {"T": form.choice_axis("T", ["300.0"])}),
    ]
    assert form.selections_from_axes(axes) == {
        "atom": ["Sr", "Ti", "O", "1", "2", "3"],
        "gap": ["fundamental", "direct", "d"],
        "spin_components": ["x~y", "y~x"],
        "mode": ["1", "2", "3"],
        "T": ["300.0"],
    }


@pytest.mark.parametrize("annotation", (int, float))
def test_from_json_rejects_numbers_too_large_for_python(annotation):
    with pytest.raises(exception.IncorrectUsage):
        form.from_json(10**400, annotation)


def test_tuple_of_any_length():
    expected = {"kind": "list", "items": {"kind": "integer"}, "length": None}
    assert form.type_spec(tuple[int, ...]) == expected
    assert form.from_json([1, 2, 3], tuple[int, ...]) == (1, 2, 3)


def test_tuple_of_different_types_is_unsupported():
    spec = form.type_spec(tuple[int, str])
    assert spec["kind"] == "unsupported"


@dataclasses.dataclass
class WithFactory:
    values: list[int] = dataclasses.field(default_factory=lambda: [1, 2])


def test_default_from_a_factory():
    (field,) = form.type_spec(WithFactory)["fields"]
    assert field["default"] == [1, 2]


def test_range_bounds_every_number_of_a_union():
    annotation = typing.Annotated[int | float | None, slicing.Range(minimum=1)]
    spec = form.type_spec(annotation)
    assert all(option["minimum"] == 1 for option in spec["options"])


def test_object_named_after_another_class_is_rejected():
    with pytest.raises(exception.IncorrectUsage):
        form.from_json({"name": "Other", "width": 1.0}, Settings)


@dataclasses.dataclass
class Validated:
    width: float

    def __post_init__(self):
        if self.width < 0:
            raise ValueError("The width must not be negative.")


def test_object_the_dataclass_refuses_is_rejected():
    with pytest.raises(exception.IncorrectUsage, match="negative"):
        form.from_json({"width": -1}, Validated)
