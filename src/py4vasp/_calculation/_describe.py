# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Describe a routine of a quantity for a tool that offers it to a user in a form.

The description is built from plain JSON types, so it can be passed on to a browser as
it is. :func:`describe` documents its layout; :mod:`py4vasp._util.form` the types of
the fields. Missing data never raises: the routine is then reported as unavailable.
:func:`call` checks JSON arguments against the same description and calls the routine.
"""

import inspect
import re
import typing
from collections.abc import Mapping
from typing import Any, Literal, NotRequired, TypedDict

from py4vasp import exception
from py4vasp._calculation import _public_quantities, _quantity_object
from py4vasp._calculation.dispatch import _availability_quantity_of
from py4vasp._raw.definition import schema
from py4vasp._third_party import graph, view
from py4vasp._util import docstring, form, slicing, tree

# routines documented under the name of the routine they forward to
_DOCUMENTED_AT = {"to_dict": "read", "__str__": "print"}


class Source(TypedDict):
    """One source of the data of a quantity, e.g. the default one or "kpoints_opt".

    Attributes
    ----------
    name : str
        The name to pass as ``source`` to select it.
    aliases : list[str]
        Other names that select the same source.
    available : bool
        Whether this calculation contains the data the routine needs from it.
    steps : int | None
        How many steps the source holds: 1 for data without steps, e.g. a single
        structure, and None if the source is unavailable or the count is unknown.
        Steps are counted from 0 when selected, while printed labels count from 1.
    """

    name: str
    aliases: list[str]
    available: bool
    steps: int | None


class Group(TypedDict):
    """Parameters of which at most one may be set, e.g. the cut a, b, or c.

    Attributes
    ----------
    exclusive : bool
        Always true: the members exclude each other.
    required : bool
        If true, exactly one member must be set; otherwise at most one.
    fields : list[str]
        The names of the members.
    """

    exclusive: bool
    required: bool
    fields: list[str]


class Field(TypedDict):
    """One parameter of a routine.

    Attributes
    ----------
    name : str
        The name of the parameter, the key in the ``arguments`` of :func:`call`.
    description : str
        What the docstring says about it; it may contain reStructuredText roles.
    required : bool
        Whether the routine cannot be called without it.
    nullable : bool
        Whether None (JSON null) is accepted; usually py4vasp then chooses itself. A
        parameter the user did not touch is left out of the arguments or sent as null
        if it is nullable; an empty string is a value, not "not set".
    default : Any, optional
        The default value in JSON types; present only if the parameter is optional.
    type : dict
        The values it takes, a dictionary with a ``kind``:

        - "boolean", "integer", "number", "text": a single value; a number may carry
          a "minimum" and a "maximum", which describe the meaningful values without
          excluding others.
        - "single_choice": one of the values in "choices".
        - "list": values of the type "items"; "length" is fixed or None.
        - "mapping": values of the type "values" under keys of the type "keys".
        - "union": a value of any of the types in "options".
        - "ellipsis": the Python ``...``, written as the string "...".
        - "object": a dataclass whose "fields" are described like these; in JSON it
          is an object of its fields, which names the class with the key "name" if
          the type allows several dataclasses.
        - "any": whatever the routine accepts.
        - "unsupported": a type without description, named by "name".
        - "selection": the selection string of the routine, see below.

        A "selection" lists its "axes", the tokens it combines besides the source,
        and its "operators", how they are combined: separators, nesting in
        parentheses, ranges, pairs, arithmetic, and assignments. Each axis has a
        "kind" and a "name": "atom" lists the "elements" and the 1-based "indices"
        [first, last]; "choice" lists "tokens", each with its "name", "aliases", and
        "description"; "pair" joins two "members", another axis, with a tilde, and
        "distinct" says whether the two must differ; "index" is a 1-based number in
        "indices" [first, last]; "assignment" is key=value for any of its "keys",
        each mapped to the axis of its values. An empty list of axes means the
        selection only chooses the source.
    group : str, optional
        The name of the exclusive group the parameter belongs to.
    """

    name: str
    description: str
    required: bool
    nullable: bool
    default: NotRequired[Any]
    type: dict[str, Any]
    group: NotRequired[str]


class Routine(TypedDict):
    """How a routine of a quantity is called and what it does.

    Attributes
    ----------
    quantity : str
        The name of the quantity, e.g. "dos" or "phonon.band".
    method : str
        The name of the routine, e.g. "to_graph".
    available : bool
        Whether any source contains the data the routine needs.
    summary : str
        The first line of the docstring, fit for a title.
    description : str
        The explanation, what the routine returns, and notes, without examples; it
        may contain reStructuredText roles.
    steps : "slice" | "single" | None
        How the quantity selects steps: a range ("slice"), one step ("single"), or
        not at all (None). Each source says how many steps it holds.
    sources : list[Source]
        The sources the routine can read from; the first one is the default.
    fields : list[Field]
        One field per parameter, in the order of the signature.
    groups : dict[str, Group]
        The exclusive groups of parameters by name.
    """

    quantity: str
    method: str
    available: bool
    summary: str
    description: str
    steps: Literal["slice", "single"] | None
    sources: list[Source]
    fields: list[Field]
    groups: dict[str, Group]


def describe(calculation, quantity: str, method: str) -> Routine:
    """Describe how *method* of *quantity* is called and what it does.

    Parameters
    ----------
    calculation : Calculation
        The calculation whose data decides the sources, steps, and tokens offered.
    quantity : str
        The name of the quantity, e.g. "dos" or "phonon.band".
    method : str
        The routine, e.g. "to_graph". ``plot`` is described as the routine it calls.

    Returns
    -------
    dict
        ``quantity`` and ``method`` as given; ``available``, whether any source has
        the data the routine needs; ``summary`` and ``description``, the docstring
        without its examples; ``steps``, "slice" if the quantity takes a range of
        steps by indexing, "single" if it takes one step, None if it has no steps;
        ``sources``, each with its ``name``, ``aliases``, whether it is ``available``,
        and its number of ``steps`` (None if unknown); ``fields``, one per parameter;
        and ``groups``, the parameters of which only one may be set.

    Raises
    ------
    IncorrectUsage
        If the quantity or the routine does not exist.
    """
    instance = _resolve_quantity(calculation, quantity)
    called = _called_routine(instance, method)
    routine = getattr(instance, called)
    documented = getattr(instance, _DOCUMENTED_AT.get(called, called))
    doc = docstring.parse(inspect.getdoc(documented))
    hints = _hints(routine)
    descriptions = {entry.name: entry.description for entry in doc.parameters}
    steps = _step_selection(type(instance))
    sources = _sources(instance, called, steps)
    parameters = [
        parameter
        for parameter in inspect.signature(routine).parameters.values()
        if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    ]
    available = any(source["available"] for source in sources)
    fields = [
        _field(instance, called, parameter, hints, descriptions, available)
        for parameter in parameters
    ]
    return {
        "quantity": quantity,
        "method": method,
        "available": available,
        "summary": doc.summary,
        "description": _description(doc),
        "steps": steps,
        "sources": sources,
        "fields": fields,
        "groups": _groups(parameters, hints),
    }


def _hints(routine):
    """The annotations of the parameters of *routine*, resolved one by one.

    A parameter without an annotation, or with one that cannot be resolved, e.g.
    because it names an optional package that is not installed, takes anything.
    """
    function = inspect.unwrap(getattr(routine, "__func__", routine))
    annotations = getattr(function, "__annotations__", {})
    hints = {}
    for name, annotation in annotations.items():
        if name == "return":
            continue
        holder = lambda: None
        holder.__annotations__ = {name: annotation}
        try:
            resolved = typing.get_type_hints(
                holder, globalns=function.__globals__, include_extras=True
            )
        except Exception:
            resolved = {name: typing.Any}
        hints[name] = resolved[name]
    return hints


def _resolve_quantity(calculation, quantity):
    known = [name for name, _ in _public_quantities()]
    if quantity not in known:
        message = (
            f"There is no quantity '{quantity}'; choose one of {', '.join(known)}."
        )
        raise exception.IncorrectUsage(message)
    return _quantity_object(calculation, quantity)


def _called_routine(instance, method):
    public = not method.startswith("_") or method == "__str__"
    if not public or not callable(getattr(instance, method, None)):
        name = type(instance).__name__
        message = f"{name} has no routine '{method}' that could be described."
        raise exception.IncorrectUsage(message)
    if method == "plot" and isinstance(instance, graph.Mixin):
        return "to_graph"
    if method == "plot" and isinstance(instance, view.Mixin):
        return "to_view"
    return method


def _description(doc):
    parts = [doc.extended_summary, *(entry.description for entry in doc.returns)]
    parts.append(doc.sections.get("Notes", ""))
    return "\n\n".join(part for part in parts if part)


def _step_selection(cls):
    getitem = cls.__dict__.get("__getitem__")
    if getitem is None:
        return None
    parameters = list(inspect.signature(getitem).parameters)[1:]
    if parameters[:1] != ["steps"]:
        return None  # the index selects something else, e.g. a source or an instance
    return getattr(cls, "_step_selection", "slice")


def _sources(instance, method, steps):
    entries = schema.sources.get(_availability_quantity_of(instance), {})
    names = [name for name, source in entries.items() if source.alias_for is None]
    try:
        availability = instance.is_available(names, method=method)
    except Exception:
        availability = {}
    return [
        _source(instance, name, entries, bool(availability.get(name)), steps)
        for name in names
    ]


def _source(instance, name, entries, available, steps):
    aliases = [alias for alias, source in entries.items() if source.alias_for == name]
    number_steps = _number_steps(instance, name, steps) if available else None
    return {
        "name": name,
        "aliases": aliases,
        "available": available,
        "steps": number_steps,
    }


def _number_steps(instance, source, steps):
    if steps is None:
        return 1
    try:
        if hasattr(instance, "_number_steps"):
            return instance._number_steps(source)
        if hasattr(instance, "number_steps"):
            return int(instance[:].number_steps(source))
    except Exception:
        pass
    return None


def _field(instance, method, parameter, hints, descriptions, available):
    description = descriptions.get(parameter.name, "")
    annotation = hints.get(parameter.name, typing.Any)
    result = form.field(parameter.name, annotation, parameter.default, description)
    if parameter.name == "selection":
        axes = _selection_axes(instance, method) if available else []
        result["type"] = {
            "kind": "selection",
            "axes": axes,
            "operators": form.OPERATORS,
        }
    return result


def _selection_axes(instance, method):
    try:
        return instance._selection_axes(method)
    except Exception:
        return []  # the data the tokens are built from cannot be read


def _groups(parameters, hints):
    groups = {}
    for parameter in parameters:
        annotation = hints.get(parameter.name, typing.Any)
        marker = form.find_marker(annotation, slicing.ExclusiveGroup)
        if marker is None:
            continue
        group = groups.setdefault(
            marker.name, {"exclusive": True, "required": False, "fields": []}
        )
        group["required"] = group["required"] or marker.required
        group["fields"].append(parameter.name)
    return groups


_STEP = re.compile(r"-?[0-9]+")


def call(
    calculation,
    quantity: str,
    method: str,
    *,
    source: str | None = None,
    steps: int | str | None = None,
    selection: str | None = None,
    arguments: dict | None = None,
):
    """Call a routine with arguments in JSON types, checked against :func:`describe`.

    Parameters
    ----------
    calculation : Calculation
        The calculation to read the data from.
    quantity, method : str
        As for :func:`describe`. ``print`` returns the text instead of printing it.
    source : str | None
        A source or one of its aliases; None uses the default source.
    steps : int | str | None
        The steps to index the quantity with, written as Python indexes a sequence
        without the brackets: "3", "-1", "2:8", or "::2".
    selection : str | None
        What the routine selects from the source, e.g. "Sr(d)".
    arguments : dict | None
        The other arguments by name, as JSON values; :func:`form.from_json` converts
        them.

    Returns
    -------
    What the routine returns. A dictionary, e.g. of ``read``, comes back as the tree
    :func:`py4vasp._util.tree.to_tree` makes of it.

    Raises
    ------
    IncorrectUsage
        If the routine does not take an argument, a required one is missing, more than
        one of an exclusive group is given, or a value does not fit its field.
    """
    description = describe(calculation, quantity, method)
    instance = _quantity_object(calculation, quantity)
    fields = {field["name"]: field for field in description["fields"]}
    combined = _combine(source, selection, description["sources"], fields)
    number_steps = _number_steps_of(source, description["sources"])
    target = _index_steps(instance, description["steps"], steps, number_steps)
    hints = _hints(getattr(instance, _called_routine(instance, method)))
    kwargs = _arguments(arguments or {}, fields, description["groups"], hints)
    if combined is not None:
        kwargs["selection"] = combined
    routine = getattr(target, "__str__" if method == "print" else method)
    result = routine(**kwargs)
    return tree.to_tree(result) if isinstance(result, Mapping) else result


def _invalid(field, message):
    "An error that names the field of the form that holds the wrong value."
    error = exception.IncorrectUsage(message)
    error.field = field
    return error


def _number_steps_of(source, sources):
    for entry in sources:
        if source is None or source in (entry["name"], *entry["aliases"]):
            return entry["steps"]  # without a source, the first one is the default
    return None


def _index_steps(instance, kind, steps, number_steps):
    if steps is None:
        return instance
    name = type(instance).__name__
    if kind is None:
        raise _invalid("steps", f"{name} has no steps to select from.")
    parsed = _parse_steps(steps)
    if kind == "single" and isinstance(parsed, slice):
        raise _invalid("steps", f"Select a single step of {name}, not '{steps}'.")
    if isinstance(parsed, int) and number_steps is not None:
        if not -number_steps <= parsed < number_steps:
            message = (
                f"The step {parsed} does not exist, the source holds {number_steps} "
                "step(s) counted from 0."
            )
            raise _invalid("steps", message)
    return instance[parsed]


def _parse_steps(steps):
    if isinstance(steps, int) and not isinstance(steps, bool):
        return steps
    parts = (
        [part.strip() for part in steps.split(":")] if isinstance(steps, str) else []
    )
    valid = all(part == "" or _STEP.fullmatch(part) for part in parts)
    if not valid or not 1 <= len(parts) <= 3 or parts == [""]:
        message = f"The steps {steps!r} are neither an index nor a slice like 2:8."
        raise _invalid("steps", message)
    if len(parts) == 1:
        return int(parts[0])
    start, stop, step = (int(part) if part else None for part in (*parts, "")[:3])
    if step == 0:
        raise _invalid("steps", f"The steps {steps!r} advance by 0.")
    return slice(start, stop, step)


def _arguments(arguments, fields, groups, hints):
    for name in arguments:
        if name == "selection" or name not in fields:
            raise _invalid(name, _unknown_argument_message(name, fields))
    for name, field in fields.items():
        if field["required"] and name != "selection" and name not in arguments:
            raise _invalid(name, f"The argument '{name}' is required.")
    for group_name, group in groups.items():
        given = [name for name in group["fields"] if arguments.get(name) is not None]
        if len(given) > 1 or (group["required"] and not given):
            members = ", ".join(group["fields"])
            quantifier = "exactly" if group["required"] else "at most"
            raise _invalid(group_name, f"Set {quantifier} one of {members}.")
    return {
        name: _convert(name, value, hints.get(name, typing.Any))
        for name, value in arguments.items()
    }


def _unknown_argument_message(name, fields):
    if name == "selection":
        return "Pass the selection as selection=, not among the other arguments."
    choices = [field for field in fields if field != "selection"]
    if not choices:
        return f"The routine takes no argument '{name}' besides the selection."
    return f"The routine takes no argument '{name}', only: {', '.join(choices)}."


def _convert(name, value, annotation):
    try:
        return form.from_json(value, annotation)
    except exception.IncorrectUsage as error:
        raise _invalid(name, f"The argument '{name}': {error}") from None


def _combine(source, selection, sources, fields):
    if source is None and not selection:
        if fields.get("selection", {}).get("required"):
            raise _invalid("selection", "The routine requires a selection.")
        return None
    if "selection" not in fields:
        field = "source" if source is not None else "selection"
        raise _invalid(field, "The routine takes neither source nor selection.")
    if source is not None:
        known = [
            name for entry in sources for name in (entry["name"], *entry["aliases"])
        ]
        if source not in known:
            message = f"The source '{source}' does not exist; use one of {known}."
            raise _invalid("source", message)
    if source is not None and selection:
        return f"{source}({selection})"
    return source if source is not None else selection
