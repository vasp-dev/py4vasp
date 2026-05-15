# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import annotations

from contextlib import contextmanager
from typing import Generic, Iterator, TypeVar

from py4vasp._util import select

T = TypeVar("T")


class DataContext(Generic[T]):
    """One matched source in a DataAccess iteration.

    Carries raw data for one resolved source along with selection metadata.
    Yielded by iterating over the result of ``DataAccess.__call__``.

    Supports two usage patterns::

        # Pattern A: explicit access
        for context in data_access(selection):
            with context.access_data() as raw:
                process(raw, context.remaining_selection)

        # Pattern B: tuple unpacking (convenience)
        for raw, context in data_access(selection):
            process(raw, context.remaining_selection)
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


class DataAccess(Generic[T]):
    """Generic callable that iterates over matched sources, yielding DataContext[T].

    Constructed by Calculation (with a real source) or via ``from_data`` for testing.
    Each call returns an iterable of ``DataContext`` — one per matched source.

    Usage inside a quantity::

        for raw, ctx in self._data(selection):
            process(raw, ctx.remaining_selection)
    """

    def __init__(self, source, quantity_name: str):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_data: T) -> DataAccess[T]:
        """Create a DataAccess that yields the given raw data directly."""
        return cls(_DataSource(raw_data), quantity_name="")

    def __call__(self, selection: str | None = None) -> Iterator[DataContext[T]]:
        """Return an iterable of DataContext[T], one per matched source."""
        if self._quantity_name:
            return self._iterate_with_resolution(selection)
        return self._iterate_passthrough(selection)

    def _iterate_passthrough(self, selection):
        """from_data path: no schema lookup, yield raw directly."""
        with self._source.access(self._quantity_name, selection=selection) as raw:
            yield DataContext(
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
                yield DataContext(
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
