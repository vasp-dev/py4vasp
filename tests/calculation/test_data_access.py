# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from py4vasp._calculation.data_access import merge_dicts, merge_graphs, merge_single
from py4vasp._third_party.graph import Graph, Series

SELECTION = "alternative"


@dataclasses.dataclass
class RawBand:
    fermi_energy: float = 0.5


@pytest.fixture
def mock_schema():
    mock = MagicMock()
    mock.selections.return_value = ("default", SELECTION)
    with patch("py4vasp._raw.definition.schema", mock):
        yield mock


class SpySource:
    """Records access calls for verification."""

    def __init__(self, raw):
        self._raw = raw
        self.calls = []

    @contextmanager
    def access(self, quantity, selection=None):
        self.calls.append({"quantity": quantity, "selection": selection})
        yield self._raw


def _make_impl(value):
    """Build a minimal Impl whose read() returns a dict and plot() returns a Graph."""

    class _Impl:
        def __init__(self, raw):
            self._raw = raw

        @classmethod
        def from_data(cls, raw):
            return cls(raw)

        def read(self):
            return {"value": self._raw.fermi_energy}

        def plot(self):
            return Graph(
                Series(x=np.array([1, 2]), y=np.array([self._raw.fermi_energy] * 2))
            )

    return _Impl


class TestMergeSingle:
    """merge_single dispatches over selections and unwraps or returns a dict."""

    def test_no_selection_returns_single_result(self):
        raw = RawBand(fermi_energy=1.0)
        source = SpySource(raw)
        Impl = _make_impl(1.0)
        result = merge_single(source, "band", None, Impl.from_data, Impl.read)
        assert result == {"value": 1.0}

    def test_single_selection_returns_unwrapped_result(self, mock_schema):
        raw = RawBand(fermi_energy=2.0)
        source = SpySource(raw)
        Impl = _make_impl(2.0)
        result = merge_single(source, "example", SELECTION, Impl.from_data, Impl.read)
        assert result == {"value": 2.0}

    def test_multiple_selections_return_dict(self, mock_schema):
        raw = RawBand(fermi_energy=3.0)
        source = SpySource(raw)
        Impl = _make_impl(3.0)
        result = merge_single(
            source, "example", f"default {SELECTION}", Impl.from_data, Impl.read
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {"default", SELECTION}
        assert result["default"] == {"value": 3.0}

    def test_extra_kwargs_forwarded_to_method(self):
        raw = RawBand(fermi_energy=1.0)
        source = SpySource(raw)
        received = {}

        class _ImplWithKwarg:
            def __init__(self, raw):
                self._raw = raw

            @classmethod
            def from_data(cls, raw):
                return cls(raw)

            def read(self, scale=1):
                received["scale"] = scale
                return {"value": self._raw.fermi_energy * scale}

        result = merge_single(
            source, "band", None, _ImplWithKwarg.from_data, _ImplWithKwarg.read, scale=5
        )
        assert received["scale"] == 5
        assert result == {"value": 5.0}

    def test_source_is_called_with_resolved_selection(self, mock_schema):
        raw = RawBand(fermi_energy=0.0)
        source = SpySource(raw)
        Impl = _make_impl(0.0)
        merge_single(source, "example", SELECTION, Impl.from_data, Impl.read)
        assert source.calls[0]["selection"] == SELECTION
        assert source.calls[0]["quantity"] == "example"


class TestMergeGraphs:
    """merge_graphs dispatches and combines Graph results via Graph.__add__."""

    def test_single_selection_returns_graph(self):
        raw = RawBand(fermi_energy=1.0)
        source = SpySource(raw)
        Impl = _make_impl(1.0)
        result = merge_graphs(source, "band", None, Impl.from_data, Impl.plot)
        assert isinstance(result, Graph)
        assert len(result) == 1

    def test_multiple_selections_merged_into_one_graph(self, mock_schema):
        raw = RawBand(fermi_energy=1.0)
        source = SpySource(raw)
        Impl = _make_impl(1.0)
        result = merge_graphs(
            source, "example", f"default {SELECTION}", Impl.from_data, Impl.plot
        )
        assert isinstance(result, Graph)
        assert len(result) == 2

    def test_extra_kwargs_forwarded_to_method(self):
        raw = RawBand(fermi_energy=1.0)
        source = SpySource(raw)
        received = {}

        class _ImplWithKwarg:
            def __init__(self, raw):
                self._raw = raw

            @classmethod
            def from_data(cls, raw):
                return cls(raw)

            def plot(self, label="default"):
                received["label"] = label
                return Graph(Series(x=np.array([1]), y=np.array([1.0]), label=label))

        merge_graphs(
            source,
            "band",
            None,
            _ImplWithKwarg.from_data,
            _ImplWithKwarg.plot,
            label="custom",
        )
        assert received["label"] == "custom"


class TestMergeDicts:
    """merge_dicts dispatches and merges dict results, prefixing keys for multiple selections."""

    def test_single_selection_returns_dict_unwrapped(self):
        raw = RawBand(fermi_energy=1.0)
        source = SpySource(raw)
        Impl = _make_impl(1.0)
        result = merge_dicts(source, "band", None, Impl.from_data, Impl.read)
        assert result == {"value": 1.0}

    def test_multiple_selections_prefix_keys(self, mock_schema):
        raw = RawBand(fermi_energy=2.0)
        source = SpySource(raw)
        Impl = _make_impl(2.0)
        result = merge_dicts(
            source, "example", f"default {SELECTION}", Impl.from_data, Impl.read
        )
        assert "value_default" in result
        assert f"value_{SELECTION}" in result
        assert result["value_default"] == 2.0

    def test_extra_kwargs_forwarded_to_method(self):
        raw = RawBand(fermi_energy=1.0)
        source = SpySource(raw)
        received = {}

        class _ImplWithKwarg:
            def __init__(self, raw):
                self._raw = raw

            @classmethod
            def from_data(cls, raw):
                return cls(raw)

            def read(self, scale=1):
                received["scale"] = scale
                return {"value": self._raw.fermi_energy * scale}

        result = merge_dicts(
            source, "band", None, _ImplWithKwarg.from_data, _ImplWithKwarg.read, scale=3
        )
        assert received["scale"] == 3
        assert result == {"value": 3.0}
