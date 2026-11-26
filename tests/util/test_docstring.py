# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import collections.abc
import inspect
import types
import typing

import pytest

from py4vasp import exception
from py4vasp._util import docstring
from py4vasp._util.docstring import Entry


def parse(text):
    return docstring.parse(inspect.cleandoc(text))


def test_summary_and_extended_summary():
    doc = parse("""Read the data into a dictionary.

        The first paragraph explains
        what you get.

        A second paragraph adds more.

        Parameters
        ----------
        selection : str
            Pick something.
        """)
    assert doc.summary == "Read the data into a dictionary."
    expected = """\
The first paragraph explains
what you get.

A second paragraph adds more."""
    assert doc.extended_summary == expected


def test_summary_may_span_several_lines():
    doc = parse("""Read the data
        into a dictionary.

        More text.""")
    assert doc.summary == "Read the data into a dictionary."
    assert doc.extended_summary == "More text."


def test_parameters():
    doc = parse("""Summary.

        Parameters
        ----------
        selection : str | None
            Pick a source.

            -   a nested list
                keeps its indentation

        width : float
            The width.
        flag
            A parameter without a type.
        """)
    description = """\
Pick a source.

-   a nested list
    keeps its indentation"""
    assert doc.parameters == (
        Entry("selection", "str | None", description),
        Entry("width", "float", "The width."),
        Entry("flag", "", "A parameter without a type."),
    )


def test_several_names_share_one_entry():
    doc = parse("""Summary.

        Parameters
        ----------
        a, b, c : float
            Exactly one of them.
        """)
    assert doc.parameters == tuple(
        Entry(name, "float", "Exactly one of them.") for name in "abc"
    )


@pytest.mark.parametrize(
    "heading",
    ("str | None = None", "str | None, optional", "str | None, default: None"),
)
def test_default_is_not_part_of_the_type(heading):
    doc = parse(f"""Summary.

        Parameters
        ----------
        selection : {heading}
            Pick a source.
        """)
    assert doc.parameters == (Entry("selection", "str | None", "Pick a source."),)


def test_variadic_parameters_are_named_without_stars():
    doc = parse("""Summary.

        Parameters
        ----------
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """)
    assert doc.parameters == (
        Entry("args", "tuple", "Positional arguments."),
        Entry("kwargs", "dict", "Keyword arguments."),
    )


def test_returns():
    doc = parse("""Summary.

        Returns
        -------
        dict
            Contains the data
            over two lines.
        count : int
            A named value.
        """)
    assert doc.returns == (
        Entry("", "dict", "Contains the data\nover two lines."),
        Entry("count", "int", "A named value."),
    )


def test_other_sections_are_kept_verbatim():
    doc = parse("""Summary.

        Notes
        -----
        Some notes.

        See Also
        --------
        py4vasp.calculation.structure : The structure.

        Examples
        --------
        >>> 1 + 1
        2
        """)
    assert doc.parameters == ()
    assert doc.returns == ()
    assert doc.sections == {
        "Notes": "Some notes.",
        "See Also": "py4vasp.calculation.structure : The structure.",
        "Examples": ">>> 1 + 1\n2",
    }


def test_underlined_text_that_is_no_section_stays_content():
    doc = parse("""Summary.

        Examples
        --------
        >>> print(table)
        BORN EFFECTIVE CHARGES
        ----------------------
        ion 1
        """)
    expected = """\
>>> print(table)
BORN EFFECTIVE CHARGES
----------------------
ion 1"""
    assert doc.sections == {"Examples": expected}


def test_empty_docstring():
    for text in (None, ""):
        doc = docstring.parse(text)
        assert doc.summary == doc.extended_summary == ""
        assert doc.parameters == doc.returns == ()
        assert doc.sections == {}


def test_repeated_section_is_an_error():
    text = """Summary.

    Notes
    -----
    First.

    Notes
    -----
    Second.
    """
    with pytest.raises(exception._Py4VaspInternalError):
        parse(text)


class _Outer:
    class Inner:
        pass


@pytest.mark.parametrize(
    "annotation, expected",
    (
        (str, "str"),
        (None, "None"),
        (type(None), "None"),
        (typing.Any, "Any"),
        (str | None, "str | None"),
        (typing.Optional[float], "float | None"),
        (typing.Union[int, str], "int | str"),
        (typing.Literal["x", "y", "auto"] | None, 'Literal["x", "y", "auto"] | None'),
        (typing.Literal[1, 2], "Literal[1, 2]"),
        (tuple[int, int, int], "tuple[int, int, int]"),
        (collections.abc.Sequence[str], "Sequence[str]"),
        (typing.Mapping[str, float] | None, "Mapping[str, float] | None"),
        (dict[str, typing.Any], "dict[str, Any]"),
        (typing.Annotated[float | None, "metadata"], "float | None"),
        (types.EllipsisType | None, "ellipsis | None"),
        # a nested class is named as the attribute that exposes it, not by its path
        (_Outer.Inner, "Inner"),
    ),
)
def test_format_type(annotation, expected):
    assert docstring.format_type(annotation) == expected
