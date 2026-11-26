# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Internal py4vasp interface for VASP's own tools.

py4vasp keeps its database interface private because it is not meant to be called by
users: ``Calculation._to_database`` and the helpers it builds on are implementation
details of the user-facing package. The tools that VASP delivers on top of py4vasp --
``vaspdb``, the structure viewer -- do need that interface, though.

This module is the sanctioned way to reach it. Every name below is a thin, documented
wrapper around a py4vasp internal, which gives us one place to see what the in-house
tools actually depend on and one place to keep working when py4vasp is refactored. In
particular, every failure is reported as a :class:`py4vasp.exception.Py4VaspError`, so a
consumer only ever has to catch py4vasp's own exception hierarchy.

The two data containers, :class:`CalculationMetaData` and :class:`DatabaseData`, are
re-exported unchanged; their fields are documented on the py4vasp dataclasses. So are
the typed dictionaries :class:`Routine`, :class:`Source`, :class:`Field`, and
:class:`Group` that describe the result of :func:`inspect`.

Nothing here is part of py4vasp's public API and none of it appears in the py4vasp
documentation. The distribution is released in lockstep with py4vasp and pins
``py4vasp-core`` exactly, so a version of this package only ever talks to the py4vasp
release it was built against.
"""

import copy
from typing import Dict, List, Optional, Tuple

from py4vasp import exception
from py4vasp._calculation import _describe, dispatch
from py4vasp._calculation._describe import Field, Group, Routine, Source
from py4vasp._raw import models
from py4vasp._raw.data import CalculationMetaData
from py4vasp._raw.data import _DatabaseData as DatabaseData
from py4vasp._util import database as _database
from py4vasp._util import tree as _tree

__version__ = "1.0.0b2"

__all__ = [
    "CalculationMetaData",
    "DatabaseData",
    "Field",
    "Group",
    "Routine",
    "Source",
    "all_database_keys",
    "call",
    "combine_database_dicts",
    "inspect",
    "parse_schema_version",
    "schema_version",
    "to_database",
    "to_tree",
]


def to_database(source):
    """Extract everything from a calculation that should be written to a database.

    This is the public form of py4vasp's private ``_to_database`` method. Pass a
    :class:`~py4vasp.Calculation` to obtain the data of the complete calculation, or a
    single quantity of it to obtain only that quantity.

    Quantities that are absent from the calculation, or that raise while being read, are
    silently dropped by py4vasp rather than propagated, so the result is always the
    subset of the data that could be extracted.

    Parameters
    ----------
    source : Calculation or quantity
        The calculation or the individual quantity to extract, e.g. ``calculation`` or
        ``calculation.structure``.

    Returns
    -------
    DatabaseData or dict
        For a calculation, a :class:`DatabaseData` with a ``metadata`` and a
        ``properties`` attribute. For a single quantity, the nested dictionary
        ``{quantity: {selection: model}}`` that would be merged into those properties.
        Not every quantity is stored in the database; those return an empty dictionary.

    Raises
    ------
    IncorrectUsage
        If *source* is not a calculation or one of its quantities, or if it is a group
        of quantities such as ``calculation.phonon``. Extract the members of a group
        individually so that no data is lost silently.

    Examples
    --------
    Extract the data of a complete calculation:

    >>> from py4vasp import demo
    >>> from vasp import backend
    >>> calculation = demo.calculation(path)
    >>> data = backend.to_database(calculation)
    >>> data.metadata.schema_version == backend.schema_version()
    True
    >>> "structure" in data.properties
    True

    Extract a single quantity:

    >>> sorted(backend.to_database(calculation.structure))
    ['structure']

    A quantity that the database does not store yields nothing:

    >>> backend.to_database(calculation.neighbor_list)
    {}
    """
    extract = getattr(source, "_to_database", None)
    if extract is not None:
        return extract()
    if isinstance(source, dispatch.Group):
        members = ", ".join(sorted(source._quantities))
        raise exception.IncorrectUsage(
            "A group of quantities is not stored in the database itself. Extract "
            f"its members individually: {members}."
        )
    if hasattr(source, "_quantity_name"):
        # py4vasp itself skips a quantity without _to_database when it assembles the
        # database data, see py4vasp._calculation._collect_to_database.
        return {}
    raise exception.IncorrectUsage(
        f"Cannot extract database data from an object of type "
        f"'{type(source).__name__}' because it is not a calculation or one of its "
        "quantities."
    )


def combine_database_dicts(*args) -> dict:
    """Deep-merge the nested dictionaries produced by :func:`to_database`.

    Overlapping keys are merged recursively as long as both values are dictionaries.
    Where neither is a dictionary, the value of the right-most argument wins. The
    arguments are left untouched.

    Parameters
    ----------
    *args : dict
        The dictionaries to combine, in increasing order of precedence.

    Returns
    -------
    dict
        A new dictionary containing the combined entries. It shares no state with the
        arguments, so merging does not corrupt the data you passed in.

    Raises
    ------
    IncorrectUsage
        If any argument is not a dictionary.
    DataMismatch
        If the same key is a dictionary in one argument and a plain value in another.
        Merging those would have to discard one of them, so it is reported instead.

    Examples
    --------
    >>> from vasp import backend
    >>> backend.combine_database_dicts({"a": {"b": 1}}, {"a": {"c": 2}})
    {'a': {'b': 1, 'c': 2}}

    The right-most argument wins where the values are not dictionaries:

    >>> backend.combine_database_dicts({"a": 1}, {"a": 2})
    {'a': 2}
    """
    for arg in args:
        if not isinstance(arg, dict):
            raise exception.IncorrectUsage(
                f"Cannot merge an object of type '{type(arg).__name__}' because it is "
                "not a dictionary."
            )
    try:
        # py4vasp merges in place and aliases the nested dictionaries of its arguments
        # into the result, so copy first to keep the promise of not touching them.
        return _database.combine_db_dicts(*(copy.deepcopy(arg) for arg in args))
    except exception._Py4VaspInternalError as error:
        raise exception.DataMismatch(str(error)) from None


def schema_version() -> str:
    """The version of the database model schema the installed py4vasp produces.

    The ``<major>.<minor>`` part is the py4vasp series. A ``+db.<counter>`` suffix marks
    an intermediate migration of the models between two py4vasp releases; a released
    schema carries no suffix. Store this alongside the data so that a consumer can tell
    which models it is looking at.

    Returns
    -------
    str
        The schema version, e.g. ``"0.11"`` or ``"0.11+db.2"``.

    Examples
    --------
    >>> from vasp import backend
    >>> backend.parse_schema_version(backend.schema_version())[1] >= 0
    True
    """
    return models.schema_version()


def parse_schema_version(version: str) -> Tuple[str, int]:
    """Split a schema version into its py4vasp series and its migration counter.

    Parameters
    ----------
    version : str
        A schema version as produced by :func:`schema_version`, i.e. either the bare
        released form ``"0.11"`` or the intermediate form ``"0.11+db.5"``.

    Returns
    -------
    tuple
        The ``"<major>.<minor>"`` series and the migration counter, which is 0 for a
        released schema.

    Raises
    ------
    IncorrectUsage
        If *version* is not a valid schema version.

    Examples
    --------
    >>> from vasp import backend
    >>> backend.parse_schema_version("0.11")
    ('0.11', 0)
    >>> backend.parse_schema_version("0.11+db.5")
    ('0.11', 5)
    """
    try:
        return models.parse_schema_version(version)
    except ValueError as error:
        raise exception.IncorrectUsage(str(error)) from None


def all_database_keys() -> (
    Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, Optional[str]]]
):
    """Enumerate every key that :func:`to_database` may produce and its data model.

    Use this to set up or validate a database schema without having to read a
    calculation first.

    Note that a group member is reported in its dotted form, ``phonon.band``, while
    :func:`to_database` emits it flattened as ``phonon_band``.

    Returns
    -------
    tuple
        A dictionary mapping the name of each model dataclass to its
        ``(attribute, type)`` pairs, and a dictionary mapping every available key
        (``quantity`` or ``quantity:selection``) to the name of its model dataclass, or
        to None where no model exists.

    Examples
    --------
    >>> from vasp import backend
    >>> models, keys = backend.all_database_keys()
    >>> keys["structure"]
    'StructureModel'
    >>> dict(models["StructureModel"])["cell_volume"]
    'Optional[float]'
    """
    return _database.get_all_possible_keys()


def inspect(calculation, quantity: str, method: str) -> Routine:
    """Describe a routine of a quantity, so that a form can offer it to a user.

    The description says where the routine reads from, which arguments it takes, what
    values they accept, and what the routine does. It contains only JSON types (str,
    int, float, bool, None, list, dict), so it can be sent to a browser unchanged.
    Missing data never raises: the routine is reported as unavailable instead, and
    the tokens of its selection are left out.

    Parameters
    ----------
    calculation : Calculation
        The calculation whose data decides the sources, steps, and selection tokens.
    quantity : str
        The name of the quantity as :meth:`py4vasp.Calculation.selections` lists it,
        e.g. ``"dos"`` or ``"phonon.band"``.
    method : str
        The routine, e.g. ``"to_graph"``. ``plot`` is described as the routine it
        calls, and ``to_dict`` and ``__str__`` with the documentation of ``read`` and
        ``print``.

    Returns
    -------
    Routine
        The description; :class:`Routine` documents every key. The parameter
        ``selection`` has a type of the kind ``"selection"``: its ``axes`` list the
        tokens besides the source, and its ``operators`` how the selection combines
        them. :class:`Field` documents the kinds of all types.

    Raises
    ------
    IncorrectUsage
        If the quantity or the routine does not exist.

    Examples
    --------
    >>> from py4vasp import demo
    >>> from vasp import backend
    >>> calculation = demo.calculation(path)
    >>> routine = backend.inspect(calculation, "density", "to_contour")
    >>> routine["summary"]
    'Generate a contour plot of the selected component of the density.'
    >>> [source["name"] for source in routine["sources"]]
    ['default', 'tau', 'all_electron']
    >>> cut = next(field for field in routine["fields"] if field["name"] == "a")
    >>> cut["type"], cut["group"]
    ({'kind': 'number', 'minimum': 0.0, 'maximum': 1.0}, 'cut')
    >>> routine["groups"]
    {'cut': {'exclusive': True, 'required': True, 'fields': ['a', 'b', 'c']}}
    """
    return _describe.describe(calculation, quantity, method)


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
    """Call a routine of a quantity with the values a user entered in its form.

    The source, the steps, and every argument are checked against the description
    :func:`inspect` returns for the same routine before anything is read, so a tool
    passes on what the user entered without building Python code from it. The values
    are JSON types: a list where a tuple is expected becomes a tuple, the string
    ``"..."`` becomes ``...``, and a JSON object becomes the dataclass its field
    describes; where a field allows several dataclasses, the object names its class
    with the key ``name``. Leave out a field the user did not touch, or send null if it
    is nullable; an empty string is a value. A range of a number only describes it, so
    a value outside of it is passed on. The tokens of the selection are not checked
    against the axes; the routine itself rejects a selection it cannot parse.

    Parameters
    ----------
    calculation : Calculation
        The calculation to read the data from.
    quantity : str
        The name of the quantity, e.g. ``"dos"`` or ``"phonon.band"``.
    method : str
        The routine, e.g. ``"to_graph"``. ``"print"`` returns the text that ``print``
        would show instead of printing it.
    source : str | None
        One of the sources :func:`inspect` lists, or one of their aliases. Without it,
        the routine reads the default source.
    steps : int | str | None
        The steps to select, written as Python indexes a sequence without the brackets:
        ``"3"``, ``"-1"``, ``"2:8"``, or ``"::2"``. Steps count from 0, while printed
        labels count from 1. Only a quantity with steps takes them, only a range if
        :func:`inspect` reports ``"slice"``, and a single step must exist in the source.
    selection : str | None
        What the routine selects from the source, e.g. ``"Sr(d)"``, combined from the
        axes :func:`inspect` lists for the field ``selection``. Choose the source with
        ``source``, not inside the selection.
    arguments : dict | None
        The values of the other fields by name.

    Returns
    -------
    -
        What the routine returns, e.g. a :class:`~py4vasp.graph.Graph`, a view, or the
        text of ``print``. A dictionary, as ``read`` returns, comes back as the tree
        :func:`to_tree` makes of it, so its arrays can be sent on as they are.

    Raises
    ------
    IncorrectUsage
        If the quantity, routine, or source does not exist, the routine takes no such
        argument, a required one is missing, more than one of an exclusive group is set,
        a value does not fit its field, or the steps do not fit the source. An error
        found before the routine runs names the field to mark as invalid in its
        attribute ``field``: the name of the argument or of the exclusive group, or
        ``"source"``, ``"steps"``, or ``"selection"``.

    Examples
    --------
    >>> from py4vasp import demo
    >>> from vasp import backend
    >>> calculation = demo.calculation(path)
    >>> arguments = {"a": 0.25, "supercell": [2, 1]}
    >>> graph = backend.call(
    ...     calculation, "density", "to_contour", selection="0", arguments=arguments
    ... )
    >>> graph.series[0].supercell
    array([2, 1])
    >>> backend.call(calculation, "structure", "print", steps="0")
    'Sr2TiO4 (step 1)\\n...'
    >>> len(backend.call(calculation, "structure", "to_view", steps="2:8").positions)
    6
    >>> data = backend.call(calculation, "structure", "read", steps="2:8")
    >>> data["positions"].shape
    (6, 7, 3)

    Only one of the cuts a, b, and c may be set:

    >>> backend.call(calculation, "density", "to_contour", arguments={"a": 0.1, "b": 0.2})
    Traceback (most recent call last):
    ...
    py4vasp.exception.IncorrectUsage: Set exactly one of a, b, c.
    """
    return _describe.call(
        calculation,
        quantity,
        method,
        source=source,
        steps=steps,
        selection=selection,
        arguments=arguments,
    )


def to_tree(value):
    """Convert data py4vasp read into dictionaries of arrays and scalars.

    A tool that shows the data sends the arrays as raw bytes and the rest as JSON.
    :func:`call` already returns the result of ``read`` in this form; use this function
    for other data, e.g. the records of :func:`to_database`.

    Parameters
    ----------
    value
        The data, e.g. a dictionary or a dataclass.

    Returns
    -------
    dict or np.ndarray or None or bool or int or float or complex or str
        A dictionary for a mapping or a dataclass instance, keyed by text. A numpy array
        of booleans, numbers, or text for a list, a tuple, or an array; bytes become
        text. A dictionary keyed by the index where the elements form no such array,
        e.g. lists of unequal length or numbers mixed with text. A Python scalar for
        anything else: a number, a boolean, text, or None as it is (a float may be NaN
        or infinite), a slice as text, and any other object as its ``str``. A slice of
        indices is counted from 1 and includes its end, as a selection writes it, e.g.
        ``"1:2"`` for the first two atoms; a slice with an open end, a negative bound,
        or a step keeps the notation of Python in brackets, e.g. ``"[3:]"``.

    Examples
    --------
    >>> import numpy as np
    >>> from vasp import backend
    >>> tree = backend.to_tree({"labels": [b"X", b"M"], "rows": [[1, 2], [3]]})
    >>> tree["labels"]
    array(['X', 'M'], dtype='<U1')
    >>> tree["rows"]
    {'0': array([1, 2]), '1': array([3])}
    >>> backend.to_tree({"count": np.int64(3), "atoms": slice(0, 2)})
    {'count': 3, 'atoms': '1:2'}
    """
    return _tree.to_tree(value)
