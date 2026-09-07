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
re-exported unchanged; their fields are documented on the py4vasp dataclasses.

Nothing here is part of py4vasp's public API and none of it appears in the py4vasp
documentation. The distribution is released in lockstep with py4vasp and pins
``py4vasp-core`` exactly, so a version of this package only ever talks to the py4vasp
release it was built against.
"""

import copy
from typing import Dict, List, Optional, Tuple

from py4vasp import exception
from py4vasp._calculation import dispatch
from py4vasp._raw import models
from py4vasp._raw.data import CalculationMetaData
from py4vasp._raw.data import _DatabaseData as DatabaseData
from py4vasp._util import database as _database

__version__ = "0.11.3"

__all__ = [
    "CalculationMetaData",
    "DatabaseData",
    "all_database_keys",
    "combine_database_dicts",
    "parse_schema_version",
    "schema_version",
    "to_database",
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
