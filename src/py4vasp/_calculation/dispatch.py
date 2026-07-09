# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Dispatch infrastructure for the new Dispatcher/Handler architecture.

Provides Source classes, dispatch functions, merge strategies, and the
@quantity registry decorator used by all quantities.
"""

import contextlib
import dataclasses
import inspect
import pathlib
import typing

import numpy as np

from py4vasp import exception
from py4vasp import raw as _raw_module
from py4vasp._raw.definition import selections as schema_selections
from py4vasp._raw.definition import unique_selections as schema_unique_selections
from py4vasp._third_party.graph import Graph
from py4vasp._util import select

_REGISTRY = {}


def quantity(name, group=None):
    """Decorator that registers a dispatcher class in the registry.

    Also injects ``from_path``, ``from_file``, and a ``_path`` property
    onto the class so that every dispatcher can be instantiated standalone.

    Parameters
    ----------
    name : str
        The attribute name for this quantity on Calculation.
    group : str | None
        If set, registers under a group namespace (e.g. "phonon").
    """

    def decorator(cls):
        # The registry key keeps a leading underscore to mark a quantity as private
        # (Calculation.__getattr__ rejects underscore-prefixed names), but the schema
        # never uses that underscore, so strip it for the access name.
        access_name = name.lstrip("_")
        if group is None:
            cls._quantity_name = access_name
        else:
            # Use the full schema name f"{group}_{name}" so that FileSource
            # can look up the correct schema entry (e.g. "electron_phonon_self_energy").
            cls._quantity_name = f"{group}_{access_name}"

        @classmethod
        def from_path(klass, path="."):
            """Create dispatcher that reads from HDF5 files at *path*."""
            return klass(source=FileSource(path))

        @classmethod
        def from_file(klass, file_name):
            """Create dispatcher that reads from a specific HDF5 file."""
            resolved = pathlib.Path(file_name).expanduser().resolve()
            return klass(source=FileSource(resolved.parent, file=file_name))

        cls.from_path = from_path
        cls.from_file = from_file

        def __repr__(self):
            return f"{type(self).__name__}.from_path({self._path!r})"

        # Preserve an explicit __repr__ defined on the class (e.g. Structure).
        if "__repr__" not in cls.__dict__:
            cls.__repr__ = __repr__

        if not isinstance(getattr(cls, "_path", None), property):
            cls._path = property(lambda self: self._source.path or pathlib.Path.cwd())

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
        with _raw_module.access(
            quantity, selection=selection, path=self._path, file=self._file
        ) as raw:
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


def _method_accepts_selection(method):
    """Check if the handler method's first positional parameter (after self) is 'selection'.

    Returns a tuple (accepts, has_default) where:
    - accepts: True if the first param is named 'selection'
    - has_default: True if that param has a default value
    """
    try:
        sig = inspect.signature(method)
        params = list(sig.parameters.values())
        non_self = [p for p in params if p.name != "self"]
        if not non_self:
            return False, False
        first = non_self[0]
        accepts = first.name == "selection" and first.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        has_default = first.default is not inspect.Parameter.empty
        return accepts, has_default
    except (ValueError, TypeError):
        return False, False


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
        Used for source routing AND automatically forwarded to the handler method
        (as remaining_selection with source prefix stripped) when the handler's first
        positional parameter is named ``selection``.
    handler_factory : callable(raw) -> Handler
        Called with the raw data object to construct a handler.
    method : unbound method reference
        The Handler method to call.
    *args, **kwargs
        Extra arguments forwarded to method(handler, [remaining_selection,] *args, **kwargs).
        Do NOT pass selection here; it is forwarded automatically when needed.

    Returns
    -------
    dict[str, result]
        Maps selection_name (or "default") to each result.
    """
    contexts = _parse_selections(quantity_name, selection)
    handler_wants_selection, selection_has_default = _method_accepts_selection(method)
    results = {}
    for ctx in contexts:
        with source.access(quantity_name, selection=ctx.selection_name) as raw:
            handler = handler_factory(raw)
            if handler_wants_selection:
                if ctx.remaining_selection is None and selection_has_default:
                    result = method(handler, *args, **kwargs)
                else:
                    result = method(handler, ctx.remaining_selection, *args, **kwargs)
            else:
                result = method(handler, *args, **kwargs)
            key = ctx.selection_name or "default"
            results[key] = result
    return results


def _result_has_data(result) -> bool:
    """Return True if *result* contains any data field that carries data.

    Dataclass instances (DB objects) are considered empty when every field is
    empty. A field is empty when it is None, or when it is a ``has_*`` flag set
    to False (such a flag only signals presence, so False means "not present").
    Empty dicts are also considered empty. Everything else is assumed to have
    data.
    """
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return any(
            _field_has_data(f.name, getattr(result, f.name))
            for f in dataclasses.fields(result)
        )
    if isinstance(result, dict):
        return bool(result)
    return True


def _field_has_data(name, value) -> bool:
    if value is None:
        return False
    if name.startswith("has_"):
        return bool(value)
    return True


_SUPPRESSED_DB_EXCEPTIONS = (
    exception.Py4VaspError,
    exception.OutdatedVaspVersion,
    exception.NoData,
    exception.FileAccessError,
    AttributeError,
    TypeError,
    ValueError,
    # a source may point to an optional external file (e.g. the structure "poscar"
    # source); its absence simply means that selection has no data for the database
    FileNotFoundError,
)


class SuppressErrorsSourceWrapper:
    def __init__(self, source):
        self._source = source

    class AccessContext:
        def __init__(self, original_context):
            self.original_context = original_context
            self.entered = False

        def __enter__(self):
            try:
                self.raw = self.original_context.__enter__()
                self.entered = True
                return self.raw
            except _SUPPRESSED_DB_EXCEPTIONS:
                return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None and issubclass(exc_type, _SUPPRESSED_DB_EXCEPTIONS):
                if self.entered:
                    try:
                        self.original_context.__exit__(exc_type, exc_val, exc_tb)
                    except Exception:
                        # Intentionally ignore teardown errors here: this wrapper suppresses
                        # known database-access exceptions and should not re-raise failures
                        # from the underlying context cleanup.
                        pass
                return True
            if self.entered:
                return self.original_context.__exit__(exc_type, exc_val, exc_tb)
            return False

    def access(self, quantity, selection=None):
        return self.AccessContext(self._source.access(quantity, selection))

    def __getattr__(self, name):
        return getattr(self._source, name)


def merge_to_database(
    source, quantity_name, handler_factory, method, *args, key_name=None, **kwargs
):
    """Collect database results across all of a quantity's sources.

    Database collection is internal and never user-selected, so this enumerates
    every unique (non-alias) source of *quantity_name* from the schema and lets
    :func:`_dispatch` handle them in one pass. Results are remapped from selection
    names to quantity-based keys: the default source maps to just the quantity
    name and every other source to ``quantity_source``. Leading underscores are
    stripped from *quantity_name* so private quantities (``_CONTCAR``,
    ``_stoichiometry``) appear without the underscore prefix.

    A derived quantity accesses the schema and raw data under one name but should
    store its results under another (e.g. ``optics`` derives from
    ``dielectric_function``). Pass *key_name* to override the name used for the
    result keys while still looking up data under *quantity_name*.

    Sources without data are dropped (errors are suppressed via
    :class:`SuppressErrorsSourceWrapper`). A source whose result merely duplicates
    the default result is dropped too, so an in-memory :class:`DataSource` (which
    ignores the selection and yields the same data for every source) collapses to
    a single ``quantity`` entry.

    Parameters
    ----------
    source : Source
        The data source (DataSource, DictSource, etc.).
    quantity_name : str
        Name used to look up data in the source and the schema. Leading
        underscores are stripped for key generation.
    handler_factory : callable(raw) -> Handler
        Called with the raw data object to construct a handler.
    method : unbound method reference
        The Handler method to call.
    *args, **kwargs
        Extra arguments forwarded to the method.

    Returns
    -------
    dict[str, dict[str, result]]
        Nested ``{quantity: {selection: result}}``. The outer key is the
        (underscore-stripped) quantity name; the inner dict is keyed by selection
        with the default source keyed ``"default"``. Empty quantities are omitted.
    """
    wrapped_source = SuppressErrorsSourceWrapper(source)
    base = (key_name or quantity_name).lstrip("_")
    selection = _all_sources_selection(quantity_name.lstrip("_"))
    raw_results = _dispatch(
        wrapped_source,
        quantity_name,
        selection,
        handler_factory,
        method,
        *args,
        **kwargs,
    )
    selections = {
        sel: result
        for sel, result in raw_results.items()
        if _result_has_data(result)
    }
    selections = _drop_duplicates_of_default(selections)
    return {base: selections} if selections else {}


def _all_sources_selection(quantity_name):
    """Return a selection string covering every unique source of *quantity_name*.

    Returns None when the quantity is unknown to the schema, so :func:`_dispatch`
    falls back to the default source only.
    """
    try:
        sources = schema_unique_selections(quantity_name)
    except exception.FileAccessError:
        return None
    return ", ".join(str(source_name) for source_name in sources)


def _drop_duplicates_of_default(selections):
    """Drop non-default selections whose result equals the ``default`` result."""
    if "default" not in selections:
        return selections
    default_result = selections["default"]
    return {
        sel: result
        for sel, result in selections.items()
        if sel == "default" or result != default_result
    }


def _substitute_remaining_selection(args, original_selection, remaining_selection):
    """Replace args[0] with remaining_selection when it equals the original dispatch selection.

    .. deprecated::
        This function is no longer used internally. The dispatch system now
        automatically forwards remaining_selection to handler methods that accept
        a ``selection`` parameter. Kept for backward compatibility.
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
