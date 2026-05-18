# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Generic, Iterator, TypeVar

from py4vasp._util import select

T = TypeVar("T")


class _DataContext(Generic[T]):
    """One matched source in a _DataAccess iteration.

    Carries raw data for one resolved source along with selection metadata.
    Internal helper used by ``_dispatch``.
    """

    __slots__ = ("_raw", "selection_name", "remaining_selection")

    def __init__(
        self,
        raw: T,
        selection_name: str | None,
        remaining_selection: str | None,
    ):
        self._raw = raw
        self.selection_name = selection_name
        self.remaining_selection = remaining_selection

    @contextmanager
    def access_data(self) -> Iterator[T]:
        """Yield the typed raw data object."""
        yield self._raw

    def __iter__(self):
        """Support ``raw, ctx = context`` tuple unpacking."""
        return iter((self._raw, self))


class _DataSource:
    """Wraps a single raw data object for testing and composition."""

    def __init__(self, raw_data):
        self._raw_data = raw_data

    @contextmanager
    def access(self, quantity: str, selection: str | None = None):
        yield self._raw_data


class _DataAccess(Generic[T]):
    """Internal helper that resolves source selections from schema and iterates contexts.

    Used by ``_dispatch`` to handle selection parsing and source access.
    """

    def __init__(self, source, quantity_name: str):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_data: T) -> _DataAccess[T]:
        """Create a _DataAccess that yields the given raw data directly."""
        return cls(_DataSource(raw_data), quantity_name="")

    def __call__(self, selection: str | None = None) -> Iterator[_DataContext[T]]:
        """Return an iterable of DataContext[T], one per matched source."""
        if self._quantity_name:
            return self._iterate_with_resolution(selection)
        return self._iterate_passthrough(selection)

    def _iterate_passthrough(self, selection):
        """from_data path: no schema lookup, yield raw directly."""
        with self._source.access(self._quantity_name, selection=selection) as raw:
            yield _DataContext(
                raw=raw,
                selection_name=None,
                remaining_selection=selection,
            )

    def _iterate_with_resolution(self, selection):
        """Source-backed path: resolve sources from schema, iterate."""
        parsed = self._parse_selection(selection)
        for source_name, remaining_parts in parsed.items():
            remaining = select.selections_to_string(remaining_parts)
            remaining = remaining if remaining else None
            with self._source.access(self._quantity_name, selection=source_name) as raw:
                yield _DataContext(
                    raw=raw,
                    selection_name=source_name,
                    remaining_selection=remaining,
                )

    def _parse_selection(self, selection):
        tree = select.Tree.from_selection(selection)
        result = {}
        for sel in tree.selections():
            source, remaining = self._find_source_in_schema(sel)
            result.setdefault(source, [])
            result[source].append(remaining)
        return result

    def _find_source_in_schema(self, selection):
        from py4vasp._raw.definition import schema

        options = schema.selections(self._quantity_name)
        for option in options:
            if select.contains(selection, option, ignore_case=True):
                return self._remove_source_token(selection, option)
        return None, list(selection)

    def _remove_source_token(self, selection, option):
        is_option = lambda part: str(part).lower() == option.lower()
        remaining = [part for part in selection if not is_option(part)]
        if len(remaining) == len(selection):
            from py4vasp import exception

            message = (
                f'py4vasp identified the source "{option}" in your selection string '
                f'"{select.selections_to_string((selection,))}". However, the source '
                f"could not be extracted from the selection. A possible reason is that "
                f"it is used in an addition or subtraction, which is not implemented."
            )
            raise exception.NotImplemented(message)
        return option.lower(), remaining


def _dispatch(source, quantity_name, selection, impl_factory, method, *args, **kwargs):
    """Resolve selections, call impl method per selection, return ``{key: result}``.

    For each resolved selection, opens the source, constructs an Impl via
    ``impl_factory(raw)``, and calls ``method(impl, *args, **kwargs)``.  The result
    is stored under the selection name (or ``"default"`` when there is no explicit
    source selection).
    """
    data_access = _DataAccess(source, quantity_name)
    results = {}
    for raw, ctx in data_access(selection=selection):
        impl = impl_factory(raw)
        result = method(impl, *args, **kwargs)
        key = ctx.selection_name if ctx.selection_name is not None else "default"
        results[key] = result
    return results


def merge_single(
    source, quantity_name, selection, impl_factory, method, *args, **kwargs
):
    """Dispatch and return the single result or a ``{selection: result}`` dict.

    - **Single selection** → result returned directly (unwrapped).
    - **Multiple selections** → ``{selection_name: result, …}`` dict.
    """
    results = _dispatch(
        source, quantity_name, selection, impl_factory, method, *args, **kwargs
    )
    if len(results) == 1:
        return next(iter(results.values()))
    return results


def merge_graphs(
    source, quantity_name, selection, impl_factory, method, *args, **kwargs
):
    """Dispatch and overlay all ``Graph`` results into a single ``Graph``.

    Uses ``Graph.__add__`` (via ``functools.reduce``) so that all series are
    combined into one graph for multi-selection plots.
    """
    results = _dispatch(
        source, quantity_name, selection, impl_factory, method, *args, **kwargs
    )
    return functools.reduce(lambda a, b: a + b, results.values())


def merge_dicts(
    source, quantity_name, selection, impl_factory, method, *args, **kwargs
):
    """Dispatch and merge ``dict`` results.

    - **Single selection** → dict returned directly.
    - **Multiple selections** → flat dict with keys prefixed by the selection
      name: ``{original_key}_{selection_name}``.
    """
    results = _dispatch(
        source, quantity_name, selection, impl_factory, method, *args, **kwargs
    )
    if len(results) == 1:
        return next(iter(results.values()))
    combined = {}
    for sel, d in results.items():
        for k, v in d.items():
            combined[f"{k}_{sel}"] = v
    return combined
