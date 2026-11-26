# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Split the numpydoc docstrings of py4vasp into their parts.

py4vasp documents its routines in the numpydoc format. Tools that present a routine to
a user, e.g. a form with one field per parameter, need the parts separately: the
summary, the explanation, the description of every parameter, and what the routine
returns. This parser covers the subset of numpydoc that py4vasp uses and needs no
dependency beyond the standard library.

Only the section names numpydoc defines start a section, so underlined text in the
output of an example stays part of that example.
"""

import dataclasses
import re
import textwrap
import types
import typing

from py4vasp import exception

SECTIONS = (
    "Parameters",
    "Other Parameters",
    "Returns",
    "Yields",
    "Receives",
    "Raises",
    "Warns",
    "Warnings",
    "See Also",
    "Notes",
    "References",
    "Examples",
    "Attributes",
    "Methods",
)
_UNDERLINE = re.compile(r"-{3,}")
_DEFAULT = re.compile(r"(,\s*optional\b|,\s*default\b|\s*=).*$")


@dataclasses.dataclass(frozen=True)
class Entry:
    "One parameter or one returned value."

    name: str
    "The name of the parameter without leading stars; empty for an unnamed value."
    type: str
    "The type as the docstring spells it, without a default; empty if not given."
    description: str
    "The indented text below the heading, dedented."


@dataclasses.dataclass(frozen=True)
class Docstring:
    "The parts of a docstring."

    summary: str = ""
    "The first paragraph joined into a single line."
    extended_summary: str = ""
    "The text between the summary and the first section."
    parameters: tuple[Entry, ...] = ()
    "Every parameter; a heading naming several parameters yields one entry each."
    returns: tuple[Entry, ...] = ()
    "Every returned value."
    sections: dict[str, str] = dataclasses.field(default_factory=dict)
    "The text of the remaining sections, e.g. Notes or Examples, by name."


def parse(text: str | None) -> Docstring:
    """Split a cleaned docstring, e.g. from :func:`inspect.getdoc`, into its parts.

    Raises
    ------
    _Py4VaspInternalError
        If a section appears twice.
    """
    if not text:
        return Docstring()
    preamble, sections = _split_sections(text.expandtabs().splitlines())
    summary, extended_summary = _split_summary(preamble)
    return Docstring(
        summary=summary,
        extended_summary=extended_summary,
        parameters=_parameters(sections.pop("Parameters", "")),
        returns=_returns(sections.pop("Returns", "")),
        sections=sections,
    )


def _split_sections(lines):
    preamble, sections = [], {}
    current = preamble
    for index, line in enumerate(lines):
        if _starts_section(lines, index):
            name = line.rstrip()
            if name in sections:
                message = f"The section '{name}' appears twice in the docstring."
                raise exception._Py4VaspInternalError(message)
            sections[name] = current = []
        elif not _underlines_section(lines, index):
            current.append(line)
    return preamble, {name: _join(body) for name, body in sections.items()}


def _starts_section(lines, index):
    next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
    return lines[index].rstrip() in SECTIONS and _UNDERLINE.fullmatch(next_line)


def _underlines_section(lines, index):
    return index > 0 and _starts_section(lines, index - 1)


def _join(lines):
    return textwrap.dedent("\n".join(lines)).strip("\n").rstrip()


def _split_summary(lines):
    text = _join(lines)
    summary, _, extended_summary = text.partition("\n\n")
    return " ".join(summary.split()), extended_summary.strip("\n")


def _parameters(text):
    entries = []
    for heading, description in _entries(text):
        names, _, type_ = heading.partition(" : ")
        type_ = _type_without_default(type_)
        for name in names.split(","):
            entries.append(Entry(name.strip().lstrip("*"), type_, description))
    return tuple(entries)


def _type_without_default(type_):
    return _DEFAULT.sub("", type_.strip())


def _returns(text):
    entries = []
    for heading, description in _entries(text):
        name, separator, type_ = heading.partition(" : ")
        if not separator:
            name, type_ = "", heading
        entries.append(Entry(name.strip(), type_.strip(), description))
    return tuple(entries)


def _entries(text):
    entries = []
    for line in text.splitlines():
        if line and not line[0].isspace():
            entries.append((line.rstrip(), []))
        elif entries:
            entries[-1][1].append(line)
    return [(heading, _join(body)) for heading, body in entries]


def format_type(annotation) -> str:
    """Spell an annotation as the heading of a parameter in a docstring writes it.

    Unions are joined with ``|``, strings in a ``Literal`` are double-quoted, and the
    metadata of an ``Annotated`` type is left out, e.g. ``Literal["x", "y"] | None``.
    """
    if annotation is None or annotation is type(None):
        return "None"
    if annotation is typing.Any:
        return "Any"
    origin, arguments = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is typing.Annotated:
        return format_type(arguments[0])
    if origin in (typing.Union, types.UnionType):
        return " | ".join(format_type(argument) for argument in arguments)
    if origin is typing.Literal:
        return f"Literal[{', '.join(map(_format_literal, arguments))}]"
    if origin is not None:
        return f"{origin.__name__}[{', '.join(map(format_type, arguments))}]"
    return getattr(annotation, "__name__", str(annotation))


def _format_literal(value):
    return f'"{value}"' if isinstance(value, str) else repr(value)
