# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import contextlib
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp._calculation.dispatch import (
    DataSource,
    DictSource,
    Group,
    SelectionContext,
    _dispatch,
    _parse_selections,
    _REGISTRY,
    merge_default,
    merge_graphs,
    merge_strings,
    quantity,
    slice_steps,
)


@contextlib.contextmanager
def _isolated_registry():
    """Restore _REGISTRY to its original state after the block."""
    saved = {k: dict(v) if isinstance(v, dict) else v for k, v in _REGISTRY.items()}
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)


class TestDataSource:
    def test_access_yields_raw_data(self):
        raw_data = {"eigenvalues": np.array([1, 2, 3])}
        source = DataSource(raw_data)
        with source.access("band") as data:
            assert data is raw_data

    def test_access_ignores_quantity_name(self):
        raw_data = {"eigenvalues": np.array([1, 2, 3])}
        source = DataSource(raw_data)
        with source.access("anything") as data:
            assert data is raw_data

    def test_access_ignores_selection(self):
        raw_data = {"eigenvalues": np.array([1, 2, 3])}
        source = DataSource(raw_data)
        with source.access("band", selection="up") as data:
            assert data is raw_data


class TestDictSource:
    def test_access_by_quantity_name(self):
        raw_band = {"eigenvalues": np.array([1, 2, 3])}
        source = DictSource({"band": raw_band})
        with source.access("band") as data:
            assert data is raw_band

    def test_access_by_quantity_and_selection(self):
        raw_band_up = {"eigenvalues": np.array([1, 2])}
        raw_band_down = {"eigenvalues": np.array([3, 4])}
        source = DictSource(
            {
                ("band", "up"): raw_band_up,
                ("band", "down"): raw_band_down,
            }
        )
        with source.access("band", selection="up") as data:
            assert data is raw_band_up
        with source.access("band", selection="down") as data:
            assert data is raw_band_down

    def test_access_falls_back_to_quantity_without_selection(self):
        raw_band = {"eigenvalues": np.array([1, 2, 3])}
        source = DictSource({"band": raw_band})
        with source.access("band", selection="nonexistent") as data:
            assert data is raw_band

    def test_access_with_none_selection_uses_quantity_name(self):
        raw_band = {"eigenvalues": np.array([1, 2, 3])}
        source = DictSource({"band": raw_band})
        with source.access("band", selection=None) as data:
            assert data is raw_band


class TestParseSelections:
    def test_schema_source_returns_source_name(self):
        with patch("py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]):
            result = _parse_selections("test_qty", "foo")
        assert result == [SelectionContext("foo", None)]

    def test_schema_source_with_remaining_in_outer_position(self):
        with patch("py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]):
            result = _parse_selections("test_qty", "foo(bar)")
        assert result == [SelectionContext("foo", "bar")]

    def test_schema_source_in_inner_position_same_result(self):
        # "bar(foo)" and "foo(bar)" must produce the same SelectionContext because
        # the schema lookup scans the whole tuple, not just position 0.
        with patch("py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]):
            result = _parse_selections("test_qty", "bar(foo)")
        assert result == [SelectionContext("foo", "bar")]

    def test_no_schema_match_becomes_remaining(self):
        with patch("py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]):
            result = _parse_selections("test_qty", "bar(baz)")
        assert result == [SelectionContext(None, "bar(baz)")]

    def test_multiple_children_are_grouped(self):
        # "foo(bar,baz)" yields two tuples from Tree that both resolve to source
        # "foo"; they must be grouped into one SelectionContext.
        with patch("py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]):
            result = _parse_selections("test_qty", "foo(bar,baz)")
        assert result == [SelectionContext("foo", "bar, baz")]

    def test_mixed_known_and_unknown_tokens(self):
        with patch("py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]):
            result = _parse_selections("test_qty", "foo, bar")
        assert result == [
            SelectionContext("foo", None),
            SelectionContext(None, "bar"),
        ]

    def test_source_matching_is_case_insensitive(self):
        with patch("py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]):
            result = _parse_selections("test_qty", "FOO(bar)")
        assert result == [SelectionContext("foo", "bar")]

    def test_operation_in_child_becomes_remaining(self):
        # "foo(bar + baz)" → source "foo", remaining is the combined expression.
        with patch("py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]):
            result = _parse_selections("test_qty", "foo(bar + baz)")
        assert result == [SelectionContext("foo", "bar + baz")]

    def test_range_notation_as_remaining(self):
        # "foo(bar:baz)" → source "foo", remaining is the range expression.
        with patch("py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]):
            result = _parse_selections("test_qty", "foo(bar:baz)")
        assert result == [SelectionContext("foo", "bar:baz")]

    def test_source_in_inner_with_range_parent(self):
        # "bar:baz(foo)" → 'foo' is the source; 'bar:baz' is the remaining range.
        with patch("py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]):
            result = _parse_selections("test_qty", "bar:baz(foo)")
        assert result == [SelectionContext("foo", "bar:baz")]

    def test_selection_context_is_named_tuple(self):
        ctx = SelectionContext("source", "remainder")
        assert ctx.selection_name == "source"
        assert ctx.remaining_selection == "remainder"


class _FakeHandler:
    """Minimal handler for testing dispatch."""

    def __init__(self, raw_data):
        self._raw_data = raw_data

    @classmethod
    def from_data(cls, raw_data):
        return cls(raw_data)

    def read(self):
        return {"value": self._raw_data["value"]}

    def read_with_args(self, scale=1):
        return {"value": self._raw_data["value"] * scale}


class TestDispatch:
    def test_dispatch_single_selection_none(self):
        raw = {"value": 42}
        source = DataSource(raw)
        results = _dispatch(
            source, "quantity", None, _FakeHandler.from_data, _FakeHandler.read
        )
        assert results == {"default": {"value": 42}}

    def test_dispatch_single_named_selection(self):
        raw = {"value": 10}
        source = DictSource({("quantity", "up"): raw})
        with patch("py4vasp._calculation.dispatch.schema_selections", return_value=["up"]):
            results = _dispatch(
                source, "quantity", "up", _FakeHandler.from_data, _FakeHandler.read
            )
        assert results == {"up": {"value": 10}}

    def test_dispatch_multiple_selections(self):
        raw_a = {"value": 1}
        raw_b = {"value": 2}
        source = DictSource({("quantity", "a"): raw_a, ("quantity", "b"): raw_b})
        with patch("py4vasp._calculation.dispatch.schema_selections", return_value=["a", "b"]):
            results = _dispatch(
                source, "quantity", "a, b", _FakeHandler.from_data, _FakeHandler.read
            )
        assert results == {"a": {"value": 1}, "b": {"value": 2}}

    def test_dispatch_forwards_extra_kwargs(self):
        raw = {"value": 5}
        source = DataSource(raw)
        results = _dispatch(
            source,
            "quantity",
            None,
            _FakeHandler.from_data,
            _FakeHandler.read_with_args,
            scale=3,
        )
        assert results == {"default": {"value": 15}}

    def test_dispatch_forwards_extra_args(self):
        raw = {"value": 5}
        source = DataSource(raw)
        results = _dispatch(
            source,
            "quantity",
            None,
            _FakeHandler.from_data,
            _FakeHandler.read_with_args,
            2,
        )
        assert results == {"default": {"value": 10}}


class TestMergeDefault:
    def test_single_selection_returns_result_directly(self):
        raw = {"value": 42}
        source = DataSource(raw)
        result = merge_default(
            source, "quantity", None, _FakeHandler.from_data, _FakeHandler.read
        )
        assert result == {"value": 42}

    def test_single_named_selection_returns_result_directly(self):
        raw = {"value": 7}
        source = DictSource({("quantity", "up"): raw})
        with patch("py4vasp._calculation.dispatch.schema_selections", return_value=["up"]):
            result = merge_default(
                source, "quantity", "up", _FakeHandler.from_data, _FakeHandler.read
            )
        assert result == {"value": 7}

    def test_multiple_selections_returns_dict(self):
        raw_a = {"value": 1}
        raw_b = {"value": 2}
        source = DictSource({("quantity", "a"): raw_a, ("quantity", "b"): raw_b})
        with patch("py4vasp._calculation.dispatch.schema_selections", return_value=["a", "b"]):
            result = merge_default(
                source, "quantity", "a, b", _FakeHandler.from_data, _FakeHandler.read
            )
        assert result == {"a": {"value": 1}, "b": {"value": 2}}

    def test_forwards_kwargs(self):
        raw = {"value": 5}
        source = DataSource(raw)
        result = merge_default(
            source,
            "quantity",
            None,
            _FakeHandler.from_data,
            _FakeHandler.read_with_args,
            scale=3,
        )
        assert result == {"value": 15}


class _GraphHandler:
    """Handler that returns Graph objects for merge_graphs tests."""

    def __init__(self, raw_data):
        self._raw_data = raw_data

    @classmethod
    def from_data(cls, raw_data):
        return cls(raw_data)

    def plot(self):
        from py4vasp._third_party.graph import Graph, Series

        x = self._raw_data["x"]
        y = self._raw_data["y"]
        return Graph(series=[Series(x=x, y=y, label="data")])


class TestMergeGraphs:
    def test_single_selection_returns_graph_directly(self):
        from py4vasp._third_party.graph import Graph

        raw = {"x": np.array([1, 2]), "y": np.array([3, 4])}
        source = DataSource(raw)
        result = merge_graphs(
            source, "quantity", None, _GraphHandler.from_data, _GraphHandler.plot
        )
        assert isinstance(result, Graph)
        assert len(result) == 1

    def test_multiple_selections_merges_graphs(self):
        from py4vasp._third_party.graph import Graph

        raw_a = {"x": np.array([1, 2]), "y": np.array([3, 4])}
        raw_b = {"x": np.array([5, 6]), "y": np.array([7, 8])}
        source = DictSource({("quantity", "a"): raw_a, ("quantity", "b"): raw_b})
        with patch("py4vasp._calculation.dispatch.schema_selections", return_value=["a", "b"]):
            result = merge_graphs(
                source, "quantity", "a, b", _GraphHandler.from_data, _GraphHandler.plot
            )
        assert isinstance(result, Graph)
        assert len(result) == 2

    def test_multiple_selections_labels_series(self):
        raw_a = {"x": np.array([1, 2]), "y": np.array([3, 4])}
        raw_b = {"x": np.array([5, 6]), "y": np.array([7, 8])}
        source = DictSource({("quantity", "a"): raw_a, ("quantity", "b"): raw_b})
        with patch("py4vasp._calculation.dispatch.schema_selections", return_value=["a", "b"]):
            result = merge_graphs(
                source, "quantity", "a, b", _GraphHandler.from_data, _GraphHandler.plot
            )
        labels = [s.label for s in result]
        assert "a" in labels
        assert "b" in labels


class _StringHandler:
    """Handler that returns strings for merge_single tests."""

    def __init__(self, raw_data):
        self._raw_data = raw_data

    @classmethod
    def from_data(cls, raw_data):
        return cls(raw_data)

    def __str__(self):
        return self._raw_data["text"]


class TestMergeStrings:
    def test_single_selection_returns_string_directly(self):
        raw = {"text": "hello"}
        source = DataSource(raw)
        result = merge_strings(
            source, "quantity", None, _StringHandler.from_data, _StringHandler.__str__
        )
        assert result == "hello"

    def test_multiple_selections_concatenates_with_newlines(self):
        raw_a = {"text": "first"}
        raw_b = {"text": "second"}
        source = DictSource({("quantity", "a"): raw_a, ("quantity", "b"): raw_b})
        with patch("py4vasp._calculation.dispatch.schema_selections", return_value=["a", "b"]):
            result = merge_strings(
                source,
                "quantity",
                "a, b",
                _StringHandler.from_data,
                _StringHandler.__str__,
            )
        assert "first" in result
        assert "second" in result


class TestSliceSteps:
    def test_none_returns_last_step(self):
        # 3 steps, each is a 2x2 matrix → shape (3, 2, 2), default_ndim=2
        data = np.arange(12).reshape(3, 2, 2)
        result = slice_steps(data, steps=None, default_ndim=2)
        np.testing.assert_array_equal(result, data[-1])

    def test_integer_returns_single_step(self):
        data = np.arange(12).reshape(3, 2, 2)
        result = slice_steps(data, steps=1, default_ndim=2)
        np.testing.assert_array_equal(result, data[1])

    def test_slice_returns_range_of_steps(self):
        data = np.arange(12).reshape(3, 2, 2)
        result = slice_steps(data, steps=slice(0, 2), default_ndim=2)
        np.testing.assert_array_equal(result, data[0:2])

    def test_no_step_dimension_passes_through(self):
        # Data has ndim == default_ndim, so there's no step dimension
        data = np.arange(6).reshape(2, 3)
        result = slice_steps(data, steps=None, default_ndim=2)
        np.testing.assert_array_equal(result, data)

    def test_no_step_dimension_with_integer_passes_through(self):
        data = np.arange(6).reshape(2, 3)
        result = slice_steps(data, steps=1, default_ndim=2)
        np.testing.assert_array_equal(result, data)

    def test_1d_data_with_steps(self):
        # 5 steps of scalar values → shape (5,), default_ndim=0
        data = np.array([10, 20, 30, 40, 50])
        result = slice_steps(data, steps=2, default_ndim=0)
        assert result == 30

    def test_1d_data_none_returns_last(self):
        data = np.array([10, 20, 30, 40, 50])
        result = slice_steps(data, steps=None, default_ndim=0)
        assert result == 50

    def test_1d_data_slice(self):
        data = np.array([10, 20, 30, 40, 50])
        result = slice_steps(data, steps=slice(1, 4), default_ndim=0)
        np.testing.assert_array_equal(result, np.array([20, 30, 40]))


class TestQuantityDecorator:
    def test_registers_top_level_quantity(self):
        with _isolated_registry():

            @quantity("test_band")
            class TestBand:
                pass

            assert "test_band" in _REGISTRY
            assert _REGISTRY["test_band"] is TestBand

    def test_stores_quantity_name_on_class(self):
        with _isolated_registry():

            @quantity("test_energy")
            class TestEnergy:
                pass

            assert TestEnergy._quantity_name == "test_energy"

    def test_registers_grouped_quantity(self):
        with _isolated_registry():

            @quantity("test_dos", group="test_phonon")
            class TestPhononDos:
                pass

            assert "test_phonon" in _REGISTRY
            assert isinstance(_REGISTRY["test_phonon"], dict)
            assert _REGISTRY["test_phonon"]["test_dos"] is TestPhononDos

    def test_multiple_quantities_in_same_group(self):
        with _isolated_registry():

            @quantity("test_dos", group="test_phonon2")
            class TestPhononDos2:
                pass

            @quantity("test_band", group="test_phonon2")
            class TestPhononBand2:
                pass

            assert _REGISTRY["test_phonon2"]["test_dos"] is TestPhononDos2
            assert _REGISTRY["test_phonon2"]["test_band"] is TestPhononBand2

    def test_decorator_returns_class_unchanged(self):
        with _isolated_registry():

            @quantity("test_unchanged")
            class TestUnchanged:
                def method(self):
                    return 42

            assert TestUnchanged().method() == 42

    def test_registry_is_clean_after_isolated_block(self):
        with _isolated_registry():

            @quantity("test_ephemeral")
            class Ephemeral:
                pass

        assert "test_ephemeral" not in _REGISTRY


class _FakeDispatcher:
    """Fake dispatcher for testing Group."""

    _quantity_name = "fake"

    def __init__(self, source, quantity_name="fake"):
        self.source = source
        self.quantity_name = quantity_name

    def read(self):
        return "read_result"


class _FakeDispatcher2:
    """Another fake dispatcher for testing Group."""

    _quantity_name = "fake2"

    def __init__(self, source, quantity_name="fake2"):
        self.source = source
        self.quantity_name = quantity_name

    def plot(self):
        return "plot_result"


class TestGroup:
    def test_attribute_access_instantiates_dispatcher(self):
        source = DataSource({"value": 1})
        group = Group(source, {"fake": _FakeDispatcher})
        result = group.fake
        assert isinstance(result, _FakeDispatcher)
        assert result.source is source

    def test_attribute_access_passes_quantity_name(self):
        source = DataSource({"value": 1})
        group = Group(source, {"fake": _FakeDispatcher})
        result = group.fake
        assert result.quantity_name == "fake"

    def test_multiple_quantities_in_group(self):
        source = DataSource({"value": 1})
        group = Group(source, {"fake": _FakeDispatcher, "fake2": _FakeDispatcher2})
        assert isinstance(group.fake, _FakeDispatcher)
        assert isinstance(group.fake2, _FakeDispatcher2)

    def test_unknown_attribute_raises_attribute_error(self):
        source = DataSource({"value": 1})
        group = Group(source, {"fake": _FakeDispatcher})
        with pytest.raises(AttributeError):
            group.nonexistent
