# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Describe the parameters of a routine as the fields of a form.

A tool that offers a py4vasp routine to a user needs to know which input each parameter
takes. This module turns the annotation of a parameter into a description built from
plain JSON types, so it can be sent to a browser as it is.

The type of a field is a dictionary with a ``kind``:

- ``boolean``, ``integer``, ``number``, ``text``: a single value. A number may carry a
  ``minimum`` and a ``maximum`` from a :class:`~py4vasp._util.slicing.Range` marker.
- ``single_choice``: one of the values listed in ``choices``.
- ``list``: values of the type ``items``; ``length`` is fixed or None.
- ``mapping``: values of the type ``values`` under keys of the type ``keys``.
- ``union``: a value of any of the types listed in ``options``.
- ``ellipsis``: the Python ``...``, written as the string ``"..."``.
- ``object``: a dataclass, whose ``fields`` are described like those of a routine.
- ``any``: whatever the routine accepts, e.g. keyword arguments it passes on.
- ``unsupported``: a type with no description, named by ``name``.

None is never part of the type; a field that accepts None is ``nullable`` instead.

:func:`from_json` turns a JSON value back into the argument a routine expects. A JSON
object for a dataclass names its class with the key ``name`` when the type allows
several dataclasses, e.g. ``{"name": "Lorentzian", "fwhm": 0.01}``.

The ``selection`` of a routine is a string in the grammar of :mod:`py4vasp._util.select`
(see :data:`OPERATORS`). Its type has the kind ``selection`` and lists ``axes``, the
tokens it combines, each a dictionary with a ``kind`` and a ``name``:

- ``atom``: the ``elements`` and the 1-based ``indices`` [first, last] of the atoms.
- ``choice``: ``tokens``, each with a ``name``, its ``aliases``, and a ``description``.
- ``pair``: two ``members``, described by an axis, joined by ``~``; with ``distinct``
  the two must differ.
- ``index``: a 1-based number in the interval ``indices`` [first, last].
- ``assignment``: ``key=value`` for any of the ``keys``, each mapped to the axis of
  its values.
"""

import collections.abc
import dataclasses
import inspect
import itertools
import types
import typing

import numpy as np
import numpy.typing

from py4vasp import exception
from py4vasp._util import docstring, select, slicing

_NONE = type(None)
_SIMPLE = {bool: "boolean", int: "integer", float: "number", str: "text"}
_LISTS = (list, collections.abc.Sequence)
_MAPPINGS = (dict, collections.abc.Mapping)


OPERATORS = {
    "separators": [",", " "],
    "nesting": ["(", ")"],
    "range": select.range_separator,
    "pair": select.pair_separator,
    "arithmetic": list(select.operators),
    "assignment": select.assignment,
}
"How the parser of :mod:`py4vasp._util.select` combines the tokens of a selection."


def token(name, aliases=(), description=""):
    "One token of a ``choice`` axis, with the other names it may be given by."
    return {
        "name": str(name),
        "aliases": [str(alias) for alias in aliases],
        "description": description,
    }


def atom_axis(elements):
    "The atoms, selected by element or by their 1-based index in the structure."
    unique = list(dict.fromkeys(str(element) for element in elements))
    return {
        "kind": "atom",
        "name": "atom",
        "elements": unique,
        "indices": [1, len(elements)],
    }


def choice_axis(name, tokens):
    "One of a fixed list of tokens; a plain string is a token without aliases."
    tokens = [token(item) if isinstance(item, str) else item for item in tokens]
    return {"kind": "choice", "name": name, "tokens": tokens}


def pair_axis(name, members, distinct=True):
    """Two members of the axis *members* joined with a tilde, e.g. Sr~Ti.

    With *distinct*, the two members must differ; otherwise a member may be paired
    with itself, e.g. Sr~Sr.
    """
    return {"kind": "pair", "name": name, "members": members, "distinct": distinct}


def index_axis(name, count):
    "A 1-based number up to *count*."
    return {"kind": "index", "name": name, "indices": [1, int(count)]}


def assignment_axis(name, keys):
    "``key=value``, where the axis under each key describes its values."
    return {"kind": "assignment", "name": name, "keys": dict(keys)}


def axes_from_selections(selections, names):
    """Turn entries of a ``selections()`` dictionary into axes.

    The entry "atom" lists the elements followed by the atom indices; every other
    entry becomes a ``choice`` axis of its tokens.
    """
    axes = []
    for name in names:
        tokens = [str(item) for item in selections[name]]
        if name == "atom":
            elements = [item for item in tokens if not item.isdecimal()]
            number_atoms = sum(item.isdecimal() for item in tokens)
            axes.append({**atom_axis(elements), "indices": [1, number_atoms]})
        else:
            axes.append(choice_axis(name, tokens))
    return axes


def selections_from_axes(axes):
    """List the tokens of the axes as a ``selections()`` dictionary does.

    Every axis becomes an entry of its name listing its tokens, aliases included: the
    atoms by element and by index, a pair as every two distinct members joined by a
    tilde, and an assignment as one entry per key with its values.
    """
    result = {}
    for axis in axes:
        if axis["kind"] == "assignment":
            result.update(selections_from_axes(axis["keys"].values()))
        else:
            result[axis["name"]] = _tokens(axis)
    return result


def _tokens(axis):
    if axis["kind"] == "atom":
        return [*axis["elements"], *_indices(axis)]
    if axis["kind"] == "choice":
        return [
            name
            for token in axis["tokens"]
            for name in (token["name"], *token["aliases"])
        ]
    if axis["kind"] == "index":
        return _indices(axis)
    if axis["kind"] == "pair":
        members = _member_names(axis["members"])
        pairs = itertools.product(members, repeat=2)
        if axis["distinct"]:
            pairs = (pair for pair in pairs if pair[0] != pair[1])
        return [f"{first}{select.pair_separator}{second}" for first, second in pairs]
    raise exception._Py4VaspInternalError(f"Unknown kind of axis {axis['kind']}.")


def _indices(axis):
    first, last = axis["indices"]
    return [str(index) for index in range(first, last + 1)]


def _member_names(axis):
    if axis["kind"] == "choice":
        return [token["name"] for token in axis["tokens"]]
    return _tokens(axis)


def field(name, annotation, default=inspect.Parameter.empty, description=""):
    """Describe one parameter of a routine or one field of a dataclass.

    Parameters
    ----------
    name : str
        The name of the parameter.
    annotation
        Its annotation, including the metadata of ``Annotated``.
    default
        Its default value; ``inspect.Parameter.empty`` if it has none.
    description : str
        What the docstring says about it.

    Returns
    -------
    dict
        The name, description, whether the field is required and nullable, the
        default if it is not required, the type, and the exclusive group if any.
    """
    required = default is inspect.Parameter.empty or default is dataclasses.MISSING
    result = {
        "name": name,
        "description": description,
        "required": required,
        "nullable": is_nullable(annotation),
    }
    if not required:
        result["default"] = to_json(default)
    result["type"] = type_spec(annotation)
    group = find_marker(annotation, slicing.ExclusiveGroup)
    if group is not None:
        result["group"] = group.name
    return result


def type_spec(annotation) -> dict:
    "Describe the values of an annotation other than None, see the module."
    base, metadata = _unwrap(annotation)
    spec = _spec(base)
    bounds = next((item for item in metadata if isinstance(item, slicing.Range)), None)
    return _bounded(spec, bounds) if bounds is not None else spec


def is_nullable(annotation) -> bool:
    "Whether a parameter with this annotation accepts None."
    base, _ = _unwrap(annotation)
    if base is typing.Any or base is _NONE or base is None:
        return True
    return _is_union(base) and _NONE in typing.get_args(base)


def find_marker(annotation, marker_type):
    "The first metadata of the ``Annotated`` annotation of the given type, or None."
    _, metadata = _unwrap(annotation)
    return next((item for item in metadata if isinstance(item, marker_type)), None)


def to_json(value):
    "Convert a default value to plain JSON types; ``...`` becomes the string ``...``."
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if value is Ellipsis:
        return "..."
    if isinstance(value, (np.generic, np.ndarray)):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [to_json(element) for element in value]
    if isinstance(value, collections.abc.Mapping):
        return {str(key): to_json(element) for key, element in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: to_json(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    return repr(value)


def _unwrap(annotation):
    if typing.get_origin(annotation) is typing.Annotated:
        base, *metadata = typing.get_args(annotation)
        return base, metadata
    return annotation, []


def _is_union(annotation):
    return typing.get_origin(annotation) in (typing.Union, types.UnionType)


def _spec(annotation):
    if annotation is numpy.typing.ArrayLike:
        # a form holds a single number or a list of them, numpy accepts much more
        return {"kind": "union", "options": [_spec(float), _spec(list[float])]}
    if _is_union(annotation):
        options = [
            _spec(arg) for arg in typing.get_args(annotation) if arg is not _NONE
        ]
        return (
            options[0] if len(options) == 1 else {"kind": "union", "options": options}
        )
    if annotation is typing.Any:
        return {"kind": "any"}
    if annotation is types.EllipsisType:
        return {"kind": "ellipsis"}
    if annotation in _SIMPLE:
        return {"kind": _SIMPLE[annotation]}
    origin, arguments = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is typing.Literal:
        return {"kind": "single_choice", "choices": list(arguments)}
    if origin is tuple:
        return _tuple_spec(annotation, arguments)
    if origin in _LISTS:
        return {"kind": "list", "items": _spec(arguments[0]), "length": None}
    if origin in _MAPPINGS:
        keys, values = arguments
        return {"kind": "mapping", "keys": _spec(keys), "values": _spec(values)}
    if dataclasses.is_dataclass(annotation):
        return _object_spec(annotation)
    return _unsupported(annotation)


def _tuple_spec(annotation, arguments):
    if len(arguments) == 2 and arguments[1] is Ellipsis:
        return {"kind": "list", "items": _spec(arguments[0]), "length": None}
    if len(set(arguments)) == 1:
        return {"kind": "list", "items": _spec(arguments[0]), "length": len(arguments)}
    return _unsupported(annotation)


def _object_spec(cls):
    hints = typing.get_type_hints(cls, include_extras=True)
    documented = docstring.parse(inspect.getdoc(cls)).parameters
    descriptions = {entry.name: entry.description for entry in documented}
    fields = [
        field(f.name, hints[f.name], _default(f), descriptions.get(f.name, ""))
        for f in dataclasses.fields(cls)
        if f.init
    ]
    return {"kind": "object", "name": cls.__name__, "fields": fields}


def _default(dataclass_field):
    if dataclass_field.default is not dataclasses.MISSING:
        return dataclass_field.default
    if dataclass_field.default_factory is not dataclasses.MISSING:
        return dataclass_field.default_factory()
    return inspect.Parameter.empty


def _unsupported(annotation):
    return {"kind": "unsupported", "name": docstring.format_type(annotation)}


def _bounded(spec, bounds):
    if spec["kind"] in ("integer", "number"):
        limits = {"minimum": bounds.minimum, "maximum": bounds.maximum}
        return {
            **spec,
            **{key: value for key, value in limits.items() if value is not None},
        }
    if spec["kind"] == "union":
        return {
            **spec,
            "options": [_bounded(option, bounds) for option in spec["options"]],
        }
    return spec


def from_json(value, annotation):
    """Convert a JSON value to the argument a parameter with this annotation takes.

    Lists become tuples where the annotation asks for one, the string ``"..."`` becomes
    ``...``, and a JSON object becomes the dataclass it describes. A
    :class:`~py4vasp._util.slicing.Range` only describes the values, so a number
    outside of it is accepted.

    Raises
    ------
    IncorrectUsage
        If the value is not one the annotation allows.
    """
    base, _ = _unwrap(annotation)
    return _from_json(value, base)


def _from_json(value, annotation):
    if annotation is typing.Any:
        return value
    if value is None:
        if is_nullable(annotation):
            return None
        raise _mismatch(value, annotation)
    if annotation is numpy.typing.ArrayLike:
        return _from_json(value, float | list[float])
    if _is_union(annotation):
        return _union_from_json(value, annotation)
    if annotation in _SIMPLE or annotation is types.EllipsisType:
        return _simple_from_json(value, annotation)
    origin, arguments = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is typing.Literal and _is_choice(value, arguments):
        return value
    if origin is tuple and isinstance(value, (list, tuple)):
        return _tuple_from_json(value, annotation, arguments)
    if origin in _LISTS and isinstance(value, (list, tuple)):
        return [_from_json(item, arguments[0]) for item in value]
    if origin in _MAPPINGS and isinstance(value, dict):
        keys, values = arguments
        return {_from_json(k, keys): _from_json(v, values) for k, v in value.items()}
    if dataclasses.is_dataclass(annotation) and isinstance(value, dict):
        return _object_from_json(value, annotation)
    raise _mismatch(value, annotation)


def _simple_from_json(value, annotation):
    try:
        return _simple_value(value, annotation)
    except OverflowError:
        raise _mismatch(value, annotation) from None


def _simple_value(value, annotation):
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if annotation is bool and isinstance(value, bool):
        return value
    if annotation is int and is_number and float(value).is_integer():
        return int(value)
    if annotation is float and is_number:
        return float(value)
    if annotation is str and isinstance(value, str):
        return value
    if annotation is types.EllipsisType and value == "...":
        return Ellipsis
    raise _mismatch(value, annotation)


def _is_choice(value, choices):
    return any(value == choice and type(value) is type(choice) for choice in choices)


def _tuple_from_json(value, annotation, arguments):
    if len(arguments) == 2 and arguments[1] is Ellipsis:
        return tuple(_from_json(item, arguments[0]) for item in value)
    if len(value) != len(arguments):
        raise _mismatch(value, annotation)
    return tuple(_from_json(item, type_) for item, type_ in zip(value, arguments))


def _union_from_json(value, annotation):
    options = [option for option in typing.get_args(annotation) if option is not _NONE]
    objects = [option for option in options if dataclasses.is_dataclass(option)]
    if isinstance(value, dict) and len(objects) > 1:
        named = next(
            (cls for cls in objects if cls.__name__ == value.get("name")), None
        )
        if named is None:
            names = ", ".join(cls.__name__ for cls in objects)
            message = f"Name the type of {value!r} with the key 'name', one of {names}."
            raise exception.IncorrectUsage(message)
        return _object_from_json(value, named)
    for option in options:
        try:
            return _from_json(value, option)
        except exception.IncorrectUsage:
            continue
    raise _mismatch(value, annotation)


def _object_from_json(value, cls):
    value = dict(value)
    if value.pop("name", cls.__name__) != cls.__name__:
        raise _mismatch(value, cls)
    hints = typing.get_type_hints(cls, include_extras=True)
    names = {f.name for f in dataclasses.fields(cls) if f.init}
    if unknown := set(value) - names:
        message = f"{cls.__name__} has no field {', '.join(sorted(unknown))}."
        raise exception.IncorrectUsage(message)
    arguments = {key: from_json(item, hints[key]) for key, item in value.items()}
    try:
        return cls(**arguments)
    except (TypeError, ValueError) as error:
        raise exception.IncorrectUsage(
            f"Cannot create {cls.__name__}: {error}"
        ) from error


def _mismatch(value, annotation):
    expected = docstring.format_type(annotation)
    return exception.IncorrectUsage(f"The value {value!r} is not a {expected}.")
