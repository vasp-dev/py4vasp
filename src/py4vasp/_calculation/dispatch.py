# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Dispatch infrastructure for the new Dispatcher/Handler architecture.

Provides Source classes, dispatch functions, merge strategies, and the
@quantity registry decorator used by all quantities.
"""

import contextlib
import pathlib
import typing

import numpy as np

from py4vasp import exception, raw as _raw_module
from py4vasp._raw.definition import selections as schema_selections
from py4vasp._third_party.graph import Graph
from py4vasp._util import select

_REGISTRY = {}


def quantity(name, group=None):
    """Decorator that registers a dispatcher class in the registry.

    Parameters
    ----------
    name : str
        The attribute name for this quantity on Calculation.
    group : str | None
        If set, registers under a group namespace (e.g. "phonon").
    """

    def decorator(cls):
        cls._quantity_name = name
        if group is None:
            _REGISTRY[name] = cls
        else:
            _REGISTRY.setdefault(group, {})[name] = cls
        return cls

    return decorator


class SelectionContext(typing.NamedTuple):
    selection_name: str | None
    remaining_selection: str | None


def _find_source_in_schema(selection, quantity_name):
    """Identify the source name and remaining parts from a parsed selection tuple.

    Mirrors base.py's _find_selection_in_schema: uses the schema to find which
    element of the tuple is the data-source identifier; the rest becomes the
    remaining parts forwarded to the handler.

    Returns (source_name, remaining_parts_list) where source_name is a str or
    None and remaining_parts_list is a list of the non-source elements.
    """
    options = schema_selections(quantity_name)
    for option in options:
        if select.contains(selection, option, ignore_case=True):
            remaining = [
                part for part in selection if str(part).lower() != option.lower()
            ]
            return option.lower(), remaining

    # No source matched: the selection is forwarded to the handler as remaining.
    return None, list(selection)


def _parse_selections(quantity_name, selection):
    """Parse a user selection into individual (source, remainder) pairs.

    Uses select.Tree to support nested selections such as "foo(bar)", where
    "foo" becomes selection_name and "bar" becomes remaining_selection.
    The schema for quantity_name is consulted to identify the source element,
    matching the approach in base.py's _find_selection_in_schema.

    Multiple Tree entries that resolve to the same source name are grouped into
    one SelectionContext (e.g. "foo(bar,baz)" → SelectionContext("foo","bar, baz")).

    Returns a list of SelectionContext named tuples.
    """
    if selection is None:
        return [SelectionContext(None, None)]
    tree = select.Tree.from_selection(selection)
    grouped = {}
    for sel in tree.selections():
        source_name, remaining = _find_source_in_schema(sel, quantity_name)
        grouped.setdefault(source_name, [])
        grouped[source_name].append(remaining)
    result = []
    for source_name, remaining_list in grouped.items():
        if remaining_list == [[]]:
            remaining_str = None
        else:
            remaining_str = select.selections_to_string(remaining_list) or None
        result.append(SelectionContext(source_name, remaining_str))
    return result


class FileSource:
    """Production source: reads raw data from HDF5 files in a directory.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory of the VASP calculation.
    file : str or pathlib.Path or None
        Specific HDF5 file to read from. If None, the schema default is used.
    """

    def __init__(self, path, file=None):
        self._path = pathlib.Path(path).expanduser().resolve()
        self._file = file

    @property
    def path(self):
        """The resolved path of the calculation directory."""
        return self._path

    @contextlib.contextmanager
    def access(self, quantity, selection=None):
        with _raw_module.access(quantity, selection=selection, path=self._path, file=self._file) as raw:
            yield raw


class DataSource:
    """Wraps a single raw data object. Ignores quantity/selection."""

    path = None

    def __init__(self, raw_data):
        self._raw_data = raw_data

    @contextlib.contextmanager
    def access(self, quantity, selection=None):
        yield self._raw_data


class DictSource:
    """Maps quantity names (with optional selection) to raw data."""

    path = None

    def __init__(self, data):
        self._data = data

    @contextlib.contextmanager
    def access(self, quantity, selection=None):
        key = (quantity, selection) if selection else quantity
        if key not in self._data:
            key = quantity
        yield self._data[key]


def _dispatch(
    source, quantity_name, selection, handler_factory, method, *args, **kwargs
):
    """Core dispatch: parse selections, call method for each, collect results.

    Parameters
    ----------
    source : Source
        The data source (DataSource, DictSource, etc.).
    quantity_name : str
        Name used to look up data in the source.
    selection : str | None
        User-provided selection string (may contain multiple comma-separated items).
    handler_factory : callable(raw) -> Handler
        Called with the raw data object to construct a handler.
    method : unbound method reference
        The Handler method to call.
    *args, **kwargs
        Extra arguments forwarded to method(handler, *args, **kwargs).

    Returns
    -------
    dict[str, result]
        Maps selection_name (or "default") to each result.
    """
    contexts = _parse_selections(quantity_name, selection)
    results = {}
    for ctx in contexts:
        with source.access(quantity_name, selection=ctx.selection_name) as raw:
            handler = handler_factory(raw)
            effective_args = _substitute_remaining_selection(args, selection, ctx.remaining_selection)
            result = method(handler, *effective_args, **kwargs)
            key = ctx.selection_name or "default"
            results[key] = result
    return results


def _substitute_remaining_selection(args, original_selection, remaining_selection):
    """Replace args[0] with remaining_selection when it equals the original dispatch selection.

    This ensures that source-level selectors (e.g. "kpoints_opt") are stripped before
    the handler receives the selection, so only the projector/content part is forwarded.
    """
    if not args or args[0] != original_selection:
        return args
    return (remaining_selection,) + args[1:]


def merge_default(
    source, quantity_name, selection, handler_factory, method, *args, **kwargs
):
    """Dispatch and merge results into a single dict.

    If a single selection is provided, the result is returned directly.
    If multiple selections are present, returns a dict keyed by selection name.
    """
    results = _dispatch(
        source, quantity_name, selection, handler_factory, method, *args, **kwargs
    )
    if len(results) == 1:
        return next(iter(results.values()))
    return results


def merge_graphs(
    source, quantity_name, selection, handler_factory, method, *args, **kwargs
):
    """Dispatch and merge Graph results into a single overlay Graph.

    If a single selection is provided, the graph is returned directly.
    If multiple selections, graphs are combined with labels from selection names.
    """
    results = _dispatch(
        source, quantity_name, selection, handler_factory, method, *args, **kwargs
    )
    if len(results) == 1:
        return next(iter(results.values()))
    merged = Graph(series=[])
    for label, graph in results.items():
        merged = merged + graph.label(label)
    return merged


def merge_strings(
    source, quantity_name, selection, handler_factory, method, *args, **kwargs
):
    """Dispatch and merge string results into a single string.

    If a single selection is provided, the string is returned directly.
    If multiple selections, strings are joined with newlines.
    """
    results = _dispatch(
        source, quantity_name, selection, handler_factory, method, *args, **kwargs
    )
    if len(results) == 1:
        return next(iter(results.values()))
    return "\n".join(results.values())


def slice_steps(data, steps, default_ndim):
    """Slice the step dimension from data.

    Parameters
    ----------
    data : np.ndarray
        Array that may have a leading step dimension.
    steps : int | slice | None
        None → last step, int → single step, slice → range.
    default_ndim : int
        The expected number of dimensions without a step axis.
        If data.ndim <= default_ndim, the data has no step dimension
        and is returned unchanged.

    Returns
    -------
    np.ndarray
        The sliced data.
    """
    data = np.asarray(data)
    if data.ndim <= default_ndim:
        return data
    if steps is None:
        return data[-1]
    try:
        return data[steps]
    except (IndexError, TypeError) as error:
        raise exception.IncorrectUsage(
            f"Error accessing step {steps!r}. Please check that it is a valid integer or slice."
        ) from error


class Group:
    """Thin namespace for nested quantities (e.g. phonon.dos, phonon.band).

    On attribute access, instantiates the dispatcher class with the source.
    """

    def __init__(self, source, quantities):
        self._source = source
        self._quantities = quantities

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            cls = self._quantities[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' has no quantity '{name}'"
            ) from None
        return cls(source=self._source, quantity_name=cls._quantity_name)
