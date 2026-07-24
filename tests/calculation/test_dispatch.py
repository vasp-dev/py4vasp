# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import contextlib
import dataclasses
import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from py4vasp import raw
from py4vasp._calculation.dispatch import (
    _REGISTRY,
    DataSource,
    DictSource,
    FileSource,
    Group,
    SelectionContext,
    _dispatch,
    _parse_selections,
    _result_has_data,
    _substitute_remaining_selection,
    data_available,
    merge_default,
    merge_graphs,
    merge_strings,
    merge_to_database,
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
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]
        ):
            result = _parse_selections("test_qty", "foo")
        assert result == [SelectionContext("foo", None)]

    def test_schema_source_with_remaining_in_outer_position(self):
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]
        ):
            result = _parse_selections("test_qty", "foo(bar)")
        assert result == [SelectionContext("foo", "bar")]

    def test_schema_source_in_inner_position_same_result(self):
        # "bar(foo)" and "foo(bar)" must produce the same SelectionContext because
        # the schema lookup scans the whole tuple, not just position 0.
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]
        ):
            result = _parse_selections("test_qty", "bar(foo)")
        assert result == [SelectionContext("foo", "bar")]

    def test_no_schema_match_becomes_remaining(self):
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]
        ):
            result = _parse_selections("test_qty", "bar(baz)")
        assert result == [SelectionContext(None, "bar(baz)")]

    def test_multiple_children_are_grouped(self):
        # "foo(bar,baz)" yields two tuples from Tree that both resolve to source
        # "foo"; they must be grouped into one SelectionContext.
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]
        ):
            result = _parse_selections("test_qty", "foo(bar,baz)")
        assert result == [SelectionContext("foo", "bar, baz")]

    def test_mixed_known_and_unknown_tokens(self):
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]
        ):
            result = _parse_selections("test_qty", "foo, bar")
        assert result == [
            SelectionContext("foo", None),
            SelectionContext(None, "bar"),
        ]

    def test_source_matching_is_case_insensitive(self):
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]
        ):
            result = _parse_selections("test_qty", "FOO(bar)")
        assert result == [SelectionContext("foo", "bar")]

    def test_operation_in_child_becomes_remaining(self):
        # "foo(bar + baz)" → source "foo", remaining is the combined expression.
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]
        ):
            result = _parse_selections("test_qty", "foo(bar + baz)")
        assert result == [SelectionContext("foo", "bar + baz")]

    def test_range_notation_as_remaining(self):
        # "foo(bar:baz)" → source "foo", remaining is the range expression.
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]
        ):
            result = _parse_selections("test_qty", "foo(bar:baz)")
        assert result == [SelectionContext("foo", "bar:baz")]

    def test_source_in_inner_with_range_parent(self):
        # "bar:baz(foo)" → 'foo' is the source; 'bar:baz' is the remaining range.
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["foo"]
        ):
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

    def read_with_selection(self, selection):
        return {"value": self._raw_data["value"], "selection": selection}

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
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["up"]
        ):
            results = _dispatch(
                source, "quantity", "up", _FakeHandler.from_data, _FakeHandler.read
            )
        assert results == {"up": {"value": 10}}

    def test_dispatch_multiple_selections(self):
        raw_a = {"value": 1}
        raw_b = {"value": 2}
        source = DictSource({("quantity", "a"): raw_a, ("quantity", "b"): raw_b})
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["a", "b"]
        ):
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

    def test_source_selector_is_stripped_before_handler_receives_selection(self):
        """Regression: when selection matches a schema source key (e.g. 'kpoints_opt'),
        the handler must receive the *remaining* selection (None here), not the raw
        source-key string.  Before the fix this caused an IncorrectUsage error because
        the projector tried to interpret 'kpoints_opt' as an orbital/atom selector."""
        received = {}

        class _RecordingHandler:
            def __init__(self, raw):
                self._raw = raw

            @classmethod
            def from_data(cls, raw):
                return cls(raw)

            def read_selection(self, selection):
                received["selection"] = selection
                return self._raw

        raw = {"value": 99}
        source = DictSource({("qty", "src"): raw})
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["src"]
        ):
            _dispatch(
                source,
                "qty",
                "src",
                _RecordingHandler.from_data,
                _RecordingHandler.read_selection,
            )
        # After stripping the source key, the handler should see None, not "src"
        assert received["selection"] is None

    def test_non_source_selection_is_forwarded_unchanged(self):
        """Plain projector/content selections (not in the schema) must pass through
        to the handler unmodified."""
        received = {}

        class _RecordingHandler:
            def __init__(self, raw):
                self._raw = raw

            @classmethod
            def from_data(cls, raw):
                return cls(raw)

            def read_selection(self, selection):
                received["selection"] = selection
                return self._raw

        raw = {"value": 7}
        source = DataSource(raw)
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["src"]
        ):
            _dispatch(
                source,
                "qty",
                "atom",
                _RecordingHandler.from_data,
                _RecordingHandler.read_selection,
            )
        assert received["selection"] == "atom"

    def test_source_with_remaining_content_selection(self):
        """'src(atom)' → source='src', handler receives remaining='atom', not 'src(atom)'."""
        received = {}

        class _RecordingHandler:
            def __init__(self, raw):
                self._raw = raw

            @classmethod
            def from_data(cls, raw):
                return cls(raw)

            def read_selection(self, selection):
                received["selection"] = selection
                return self._raw

        raw = {"value": 3}
        source = DictSource({("qty", "src"): raw})
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["src"]
        ):
            _dispatch(
                source,
                "qty",
                "src(atom)",
                _RecordingHandler.from_data,
                _RecordingHandler.read_selection,
            )
        assert received["selection"] == "atom"

    def test_selection_not_forwarded_when_handler_has_no_selection_param(self):
        """If the handler method has no `selection` parameter, dispatch must not
        forward it — only source routing happens."""
        raw = {"value": 42}
        source = DataSource(raw)
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["src"]
        ):
            result = _dispatch(
                source,
                "quantity",
                "something",
                _FakeHandler.from_data,
                _FakeHandler.read,
            )
        assert result == {"default": {"value": 42}}

    def test_handler_with_selection_receives_remaining_selection(self):
        """If the handler method accepts `selection`, dispatch auto-forwards
        the remaining_selection."""
        raw = {"value": 5}
        source = DataSource(raw)
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["src"]
        ):
            result = _dispatch(
                source,
                "quantity",
                None,
                _FakeHandler.from_data,
                _FakeHandler.read_with_selection,
            )
        assert result == {"default": {"value": 5, "selection": None}}


@contextlib.contextmanager
def _patch_sources(sources):
    """Patch both the source enumeration and the parser to *sources*.

    ``merge_to_database`` enumerates ``schema_unique_selections`` to decide which
    sources to collect; ``_dispatch`` consults ``schema_selections`` to identify
    the source element while parsing. Patching both keeps the unit tests isolated
    from the real schema.
    """
    with (
        patch(
            "py4vasp._calculation.dispatch.schema_unique_selections",
            return_value=sources,
        ),
        patch("py4vasp._calculation.dispatch.schema_selections", return_value=sources),
    ):
        yield


class TestDispatchToDatabase:
    def test_default_selection_uses_quantity_name_as_key(self):
        raw = {"value": 42}
        source = DataSource(raw)
        with _patch_sources(["default"]):
            result = merge_to_database(
                source, "energy", _FakeHandler.from_data, _FakeHandler.read
            )
        assert result == {"energy": {"default": {"value": 42}}}

    def test_named_selection_becomes_inner_key(self):
        raw = {"value": 10}
        source = DictSource({("band", "kpoints_opt"): raw})
        with _patch_sources(["kpoints_opt"]):
            result = merge_to_database(
                source, "band", _FakeHandler.from_data, _FakeHandler.read
            )
        assert result == {"band": {"kpoints_opt": {"value": 10}}}

    def test_multiple_selections_produce_distinct_inner_keys(self):
        raw_a = {"value": 1}
        raw_b = {"value": 2}
        source = DictSource({("dos", "a"): raw_a, ("dos", "b"): raw_b})
        with _patch_sources(["a", "b"]):
            result = merge_to_database(
                source, "dos", _FakeHandler.from_data, _FakeHandler.read
            )
        assert result == {"dos": {"a": {"value": 1}, "b": {"value": 2}}}

    def test_leading_underscore_stripped_from_quantity_name(self):
        raw = {"value": 7}
        source = DataSource(raw)
        with _patch_sources(["default"]):
            result = merge_to_database(
                source, "_stoichiometry", _FakeHandler.from_data, _FakeHandler.read
            )
        assert "stoichiometry" in result
        assert "_stoichiometry" not in result

    def test_leading_underscore_stripped_with_named_selection(self):
        raw = {"value": 3}
        source = DictSource({("_CONTCAR", "sel"): raw})
        with _patch_sources(["sel"]):
            result = merge_to_database(
                source, "_CONTCAR", _FakeHandler.from_data, _FakeHandler.read
            )
        assert result == {"CONTCAR": {"sel": {"value": 3}}}

    def test_default_selection_is_inner_key(self):
        raw = {"value": 5}
        source = DataSource(raw)
        with _patch_sources(["default"]):
            result = merge_to_database(
                source, "force", _FakeHandler.from_data, _FakeHandler.read
            )
        assert set(result) == {"force"}
        assert set(result["force"]) == {"default"}

    def test_duplicate_source_results_collapse_to_default(self):
        # an in-memory DataSource yields the same data for every source, so the
        # non-default selections must collapse into the single default inner key
        raw = {"value": 99}
        source = DataSource(raw)
        with _patch_sources(["default", "final", "exciton"]):
            result = merge_to_database(
                source, "structure", _FakeHandler.from_data, _FakeHandler.read
            )
        assert result == {"structure": {"default": {"value": 99}}}

    def test_forwards_extra_kwargs_to_handler(self):
        raw = {"value": 4}
        source = DataSource(raw)
        with _patch_sources(["default"]):
            result = merge_to_database(
                source,
                "energy",
                _FakeHandler.from_data,
                _FakeHandler.read_with_args,
                scale=3,
            )
        assert result == {"energy": {"default": {"value": 12}}}


@dataclasses.dataclass
class _AllNoneDB:
    x: float = None
    y: float = None


@dataclasses.dataclass
class _PartialDB:
    x: float = None
    y: float = 1.0


@dataclasses.dataclass
class _HasFlagDB:
    has_value: bool = False
    value: float = None


class _AllNoneHandler:
    def __init__(self, raw_data):
        pass

    @classmethod
    def from_data(cls, raw_data):
        return cls(raw_data)

    def to_database(self):
        return _AllNoneDB()


class _EmptyDictHandler:
    def __init__(self, raw_data):
        pass

    @classmethod
    def from_data(cls, raw_data):
        return cls(raw_data)

    def to_database(self):
        return {}


class TestResultHasData:
    def test_all_none_dataclass_has_no_data(self):
        assert not _result_has_data(_AllNoneDB())

    def test_partially_filled_dataclass_has_data(self):
        assert _result_has_data(_PartialDB())

    def test_fully_filled_dataclass_has_data(self):
        assert _result_has_data(_PartialDB(x=0.0, y=1.0))

    def test_zero_value_counts_as_data(self):
        assert _result_has_data(_AllNoneDB(x=0.0))

    def test_empty_dict_has_no_data(self):
        assert not _result_has_data({})

    def test_non_empty_dict_has_data(self):
        assert _result_has_data({"key": "value"})

    def test_false_has_flag_counts_as_no_data(self):
        assert not _result_has_data(_HasFlagDB(has_value=False))

    def test_true_has_flag_counts_as_data(self):
        assert _result_has_data(_HasFlagDB(has_value=True))

    def test_dataclass_class_itself_treated_as_has_data(self):
        assert _result_has_data(_AllNoneDB)

    def test_non_dataclass_non_dict_treated_as_has_data(self):
        assert _result_has_data("some string")
        assert _result_has_data(42)


class TestMergeToDatabaseFilter:
    def test_all_none_dataclass_result_excluded(self):
        source = DataSource({"value": 42})
        result = merge_to_database(
            source, "quantity", _AllNoneHandler.from_data, _AllNoneHandler.to_database
        )
        assert result == {}

    def test_empty_dict_result_excluded(self):
        source = DataSource({"value": 42})
        result = merge_to_database(
            source,
            "quantity",
            _EmptyDictHandler.from_data,
            _EmptyDictHandler.to_database,
        )
        assert result == {}

    def test_partial_dataclass_result_included(self):
        class _PartialHandler:
            def __init__(self, raw_data):
                pass

            @classmethod
            def from_data(cls, raw_data):
                return cls(raw_data)

            def to_database(self):
                return _PartialDB()

        source = DataSource({})
        result = merge_to_database(
            source, "quantity", _PartialHandler.from_data, _PartialHandler.to_database
        )
        assert "quantity" in result
        assert isinstance(result["quantity"]["default"], _PartialDB)

    def test_non_empty_dict_result_included(self):
        source = DataSource({"value": 1})
        result = merge_to_database(
            source, "quantity", _FakeHandler.from_data, _FakeHandler.read
        )
        assert result == {"quantity": {"default": {"value": 1}}}


class TestSubstituteRemainingSelection:
    def test_replaces_first_arg_when_it_matches_original(self):
        assert _substitute_remaining_selection(("src",), "src", None) == (None,)

    def test_replaces_with_non_none_remaining(self):
        assert _substitute_remaining_selection(("src(a)",), "src(a)", "a") == ("a",)

    def test_leaves_trailing_args_intact(self):
        result = _substitute_remaining_selection(("src", 1.0), "src", None)
        assert result == (None, 1.0)

    def test_no_substitution_when_args_empty(self):
        assert _substitute_remaining_selection((), "src", None) == ()

    def test_no_substitution_when_args_differ(self):
        # args[0] is a different value — leave it alone
        result = _substitute_remaining_selection(("other",), "src", None)
        assert result == ("other",)

    def test_none_selection_is_a_noop(self):
        # Both original and remaining are None — result unchanged
        result = _substitute_remaining_selection((None,), None, None)
        assert result == (None,)


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
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["up"]
        ):
            result = merge_default(
                source, "quantity", "up", _FakeHandler.from_data, _FakeHandler.read
            )
        assert result == {"value": 7}

    def test_multiple_selections_returns_dict(self):
        raw_a = {"value": 1}
        raw_b = {"value": 2}
        source = DictSource({("quantity", "a"): raw_a, ("quantity", "b"): raw_b})
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["a", "b"]
        ):
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
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["a", "b"]
        ):
            result = merge_graphs(
                source, "quantity", "a, b", _GraphHandler.from_data, _GraphHandler.plot
            )
        assert isinstance(result, Graph)
        assert len(result) == 2

    def test_multiple_selections_labels_series(self):
        raw_a = {"x": np.array([1, 2]), "y": np.array([3, 4])}
        raw_b = {"x": np.array([5, 6]), "y": np.array([7, 8])}
        source = DictSource({("quantity", "a"): raw_a, ("quantity", "b"): raw_b})
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["a", "b"]
        ):
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
        with patch(
            "py4vasp._calculation.dispatch.schema_selections", return_value=["a", "b"]
        ):
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

    def test_stores_full_quantity_name_for_grouped_quantity(self):
        with _isolated_registry():

            @quantity("test_dos", group="test_phonon")
            class TestPhononDos:
                pass

            assert TestPhononDos._quantity_name == "test_phonon_test_dos"

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

    def test_attribute_access_passes_full_quantity_name_for_decorated_group(self):
        # Regression test: @quantity(name, group=group) must store the full
        # "group_name" so that Group passes the correct schema key to the
        # dispatcher constructor, not just the short member name.
        with _isolated_registry():

            @quantity("test_dos", group="test_phonon")
            class TestPhononDos:
                def __init__(self, source, quantity_name="test_phonon_test_dos"):
                    self.source = source
                    self.quantity_name = quantity_name

            source = DataSource({"value": 1})
            group = Group(source, _REGISTRY["test_phonon"])
            result = group.test_dos
            assert result.quantity_name == "test_phonon_test_dos"

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


class TestFileSource:
    def test_path_returns_resolved_pathlib_path(self, tmp_path):
        source = FileSource(tmp_path)
        assert source.path == tmp_path.resolve()
        assert isinstance(source.path, pathlib.Path)

    def test_path_resolves_relative_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        source = FileSource(".")
        assert source.path == tmp_path.resolve()

    def test_access_delegates_to_raw_access(self, tmp_path):
        raw_data = object()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=raw_data)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        source = FileSource(tmp_path)
        with patch(
            "py4vasp._calculation.dispatch._raw_module.access"
        ) as mock_raw_access:
            mock_raw_access.return_value = mock_ctx
            with source.access("band") as data:
                assert data is raw_data
            mock_raw_access.assert_called_once_with(
                "band", selection=None, path=source.path, file=None
            )

    def test_access_forwards_selection(self, tmp_path):
        raw_data = object()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=raw_data)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        source = FileSource(tmp_path)
        with patch(
            "py4vasp._calculation.dispatch._raw_module.access"
        ) as mock_raw_access:
            mock_raw_access.return_value = mock_ctx
            with source.access("band", selection="kpoints_opt") as data:
                pass
            mock_raw_access.assert_called_once_with(
                "band", selection="kpoints_opt", path=source.path, file=None
            )

    def test_access_forwards_file_kwarg(self, tmp_path):
        raw_data = object()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=raw_data)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        file_name = tmp_path / "vaspout.h5"
        source = FileSource(tmp_path, file=file_name)
        with patch(
            "py4vasp._calculation.dispatch._raw_module.access"
        ) as mock_raw_access:
            mock_raw_access.return_value = mock_ctx
            with source.access("energy") as data:
                pass
            mock_raw_access.assert_called_once_with(
                "energy", selection=None, path=source.path, file=file_name
            )


class TestSourcePathProperty:
    def test_data_source_path_is_none(self):
        source = DataSource(object())
        assert source.path is None

    def test_dict_source_path_is_none(self):
        source = DictSource({})
        assert source.path is None


class TestDataAvailable:
    def test_required_data_available(self, raw_data):
        source = DataSource(raw_data.density("Sr2TiO4"))
        # density has only required fields (structure link + charge)
        assert data_available(source, "density")

    def test_missing_required_field(self, raw_data):
        density = raw_data.density("Sr2TiO4")
        density.charge = raw.VaspData(None)
        source = DataSource(density)
        assert not data_available(source, "density")

    def test_optional_field_enforced_by_name(self, raw_data):
        band = raw_data.phonon_band("default")
        band.primitive_positions = raw.VaspData(None)
        source = DataSource(band)
        # an optional field is ignored unless its name is explicitly enforced
        assert data_available(source, "phonon_band")
        assert not data_available(
            source, "phonon_band", enforce_optional=["primitive_positions"]
        )

    def test_optional_present_when_enforced_by_name(self, raw_data):
        source = DataSource(raw_data.phonon_band("default"))
        assert data_available(
            source, "phonon_band", enforce_optional=["primitive_positions"]
        )

    def test_linked_optional_enforced_by_name(self, raw_data):
        # density links to structure, whose optional symmetry is absent in the demo
        source = DataSource(raw_data.density("Sr2TiO4"))
        assert data_available(source, "density")
        assert not data_available(
            source, "density", enforce_optional_linked=["symmetry"]
        )

    def test_missing_linked_required_data(self, raw_data):
        density = raw_data.density("Sr2TiO4")
        density.structure.positions = raw.VaspData(None)
        source = DataSource(density)
        assert not data_available(source, "density")

    def test_missing_file_returns_false(self, tmp_path):
        source = FileSource(tmp_path)
        assert not data_available(source, "density")


class TestIsAvailableInjected:
    def _calc(self, tmp_path):
        from py4vasp import demo

        return demo.calculation(tmp_path / "example")

    def test_injected_on_all_quantities(self, tmp_path):
        calc = self._calc(tmp_path)
        assert calc.density.is_available("default") is True
        assert calc.structure.is_available("default") is True
        assert calc.energy.is_available("default") is True

    def test_returns_false_when_quantity_absent(self, tmp_path):
        calc = self._calc(tmp_path)
        # born_effective_charge is not part of the default demo data
        assert calc.born_effective_charge.is_available("default") is False

    def test_method_argument_is_ignored_by_default(self, tmp_path):
        calc = self._calc(tmp_path)
        assert calc.density.is_available(
            "default", method="to_view"
        ) == calc.density.is_available("default")

    def test_default_does_not_enforce_optional(self, tmp_path):
        calc = self._calc(tmp_path)
        # energy uses the default _is_available, which lives in dispatch and so can be
        # observed through the patched is_available_raw (density has a custom one).
        with patch(
            "py4vasp._calculation.dispatch.is_available_raw", return_value=True
        ) as mock_available:
            calc.energy.is_available("default")
        _, kwargs = mock_available.call_args
        assert kwargs.get("enforce_optional", ()) == ()

    def test_selection_none_returns_dict_over_sources(self, tmp_path):
        calc = self._calc(tmp_path)
        result = calc.structure.is_available()
        assert isinstance(result, dict)
        # structure exposes several sources; the primary one is available
        assert result["default"] is True
        assert all(isinstance(value, bool) for value in result.values())

    def test_selection_none_uses_method_per_source(self, tmp_path):
        calc = self._calc(tmp_path)
        # density has no magnetization in the default demo, so to_quiver is unavailable
        result = calc.density.is_available(method="to_quiver")
        assert isinstance(result, dict)
        assert result["default"] is False

    def test_public_is_available_hides_enforce_optional(self):
        import inspect

        from py4vasp._calculation.dispatch import is_available

        parameters = set(inspect.signature(is_available).parameters)
        assert parameters == {"self", "selection", "method"}


class TestIsAvailableSourceResolution:
    def _calc(self, tmp_path):
        from py4vasp import demo

        return demo.calculation(tmp_path / "example")

    def test_single_nondefault_source_resolves(self, tmp_path):
        # current_density's only schema source is "nmr" (no "default"); is_available
        # must resolve it rather than fail on the missing "default" source.
        calc = self._calc(tmp_path)
        assert calc.current_density.is_available("nmr") is True
        assert calc.current_density.is_available() == {"nmr": True}
