# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from py4vasp import exception
from py4vasp._calculation.data_access import DataAccess, DataContext

SELECTION = "alternative"


@dataclasses.dataclass
class RawBand:
    fermi_energy: float = 0.5


@dataclasses.dataclass
class RawStructure:
    elements: list = dataclasses.field(default_factory=list)


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


class TestDataContext:
    def test_tuple_unpacking(self):
        raw = RawBand()
        ctx = DataContext(raw, selection_name="src", remaining_selection="rem")
        unpacked_raw, unpacked_ctx = ctx
        assert unpacked_raw is raw
        assert unpacked_ctx is ctx

    def test_access_data_yields_raw(self):
        raw = RawBand()
        ctx = DataContext(raw, selection_name=None, remaining_selection=None)
        with ctx.access_data() as raw_data:
            assert raw_data is raw

    def test_selection_attributes(self):
        ctx = DataContext(
            RawBand(), selection_name="kpoints_opt", remaining_selection="Sr p"
        )
        assert ctx.selection_name == "kpoints_opt"
        assert ctx.remaining_selection == "Sr p"


class TestFromData:
    """DataAccess.from_data(raw) wraps raw data for direct access."""

    def test_yields_one_context(self):
        raw = RawBand()
        contexts = list(DataAccess.from_data(raw)())
        assert len(contexts) == 1

    def test_access_data_yields_raw_object(self):
        raw = RawBand()
        for context in DataAccess.from_data(raw)():
            with context.access_data() as raw_data:
                assert raw_data is raw

    def test_selection_name_is_none(self):
        raw = RawBand()
        for context in DataAccess.from_data(raw)():
            assert context.selection_name is None

    def test_remaining_selection_is_none_without_selection(self):
        raw = RawBand()
        for context in DataAccess.from_data(raw)():
            assert context.remaining_selection is None

    def test_selection_passed_as_remaining(self):
        raw = RawBand()
        for context in DataAccess.from_data(raw)(selection="Sr p"):
            assert context.remaining_selection == "Sr p"
            assert context.selection_name is None

    def test_tuple_unpacking_yields_raw_and_context(self):
        raw = RawBand()
        for raw_data, ctx in DataAccess.from_data(raw)():
            assert raw_data is raw
            assert isinstance(ctx, DataContext)
            assert ctx.selection_name is None

    def test_each_call_creates_fresh_iterator(self):
        raw = RawBand()
        access = DataAccess.from_data(raw)
        assert len(list(access())) == 1
        assert len(list(access())) == 1


class TestSourceBacked:
    """DataAccess(source, quantity_name) delegates to the source."""

    def test_calls_source_with_quantity_name(self):
        spy = SpySource(RawBand())
        list(DataAccess(spy, "band")())
        assert spy.calls[0]["quantity"] == "band"

    def test_yields_raw_from_source(self):
        raw = RawBand()
        spy = SpySource(raw)
        for raw_data, _ in DataAccess(spy, "band")():
            assert raw_data is raw

    def test_no_selection_passes_none_to_source(self):
        spy = SpySource(RawBand())
        list(DataAccess(spy, "band")())
        assert spy.calls[0]["selection"] is None


class TestSourceResolution:
    """DataAccess resolves source names from the schema and strips them from selection."""

    def test_single_source_recognized(self, mock_schema):
        raw = RawBand()
        spy = SpySource(raw)
        contexts = list(DataAccess(spy, "example")(selection=SELECTION))
        assert len(contexts) == 1
        _, ctx = contexts[0]
        assert ctx.selection_name == SELECTION
        assert ctx.remaining_selection is None
        assert spy.calls[0]["selection"] == SELECTION

    def test_source_stripped_from_compound_selection(self, mock_schema):
        raw = RawBand()
        spy = SpySource(raw)
        contexts = list(DataAccess(spy, "example")(selection=f"{SELECTION}(Sr p)"))
        assert len(contexts) == 1
        _, ctx = contexts[0]
        assert ctx.selection_name == SELECTION
        assert ctx.remaining_selection == "Sr, p"
        assert spy.calls[0]["selection"] == SELECTION

    def test_non_source_tokens_pass_through(self, mock_schema):
        raw = RawBand()
        spy = SpySource(raw)
        contexts = list(DataAccess(spy, "example")(selection="Sr p"))
        assert len(contexts) == 1
        _, ctx = contexts[0]
        assert ctx.selection_name is None
        assert ctx.remaining_selection == "Sr, p"
        assert spy.calls[0]["selection"] is None

    def test_whitespace_and_case_normalization(self, mock_schema):
        raw = RawBand()
        spy = SpySource(raw)
        selection = f"  {SELECTION.upper()}  "
        contexts = list(DataAccess(spy, "example")(selection=selection))
        _, ctx = contexts[0]
        assert ctx.selection_name == SELECTION
        assert spy.calls[0]["selection"] == SELECTION

    def test_multiple_sources_yield_multiple_contexts(self, mock_schema):
        raw = RawBand()
        spy = SpySource(raw)
        contexts = list(DataAccess(spy, "example")(selection=f"default {SELECTION}"))
        assert len(contexts) == 2
        names = {ctx.selection_name for _, ctx in contexts}
        assert names == {"default", SELECTION}

    def test_mixed_source_and_non_source(self, mock_schema):
        raw = RawBand()
        spy = SpySource(raw)
        contexts = list(DataAccess(spy, "example")(selection=f"foo {SELECTION}(bar)"))
        assert len(contexts) == 2
        by_name = {ctx.selection_name: ctx for _, ctx in contexts}
        assert by_name[None].remaining_selection == "foo"
        assert by_name[SELECTION].remaining_selection == "bar"

    def test_from_data_skips_schema_lookup(self, mock_schema):
        raw = RawBand()
        for _, ctx in DataAccess.from_data(raw)(selection=SELECTION):
            assert ctx.selection_name is None
            assert ctx.remaining_selection == SELECTION
        mock_schema.selections.assert_not_called()

    def test_no_selection_yields_default_source(self, mock_schema):
        raw = RawBand()
        spy = SpySource(raw)
        contexts = list(DataAccess(spy, "example")())
        assert len(contexts) == 1
        _, ctx = contexts[0]
        assert ctx.selection_name is None
        assert ctx.remaining_selection is None


class TestErrorHandling:
    """DataAccess raises appropriate errors for invalid selections."""

    @pytest.mark.parametrize("operator", ["+", "-"])
    def test_operations_with_source_raise_not_implemented(self, operator, mock_schema):
        spy = SpySource(RawBand())
        with pytest.raises(exception.NotImplemented):
            list(
                DataAccess(spy, "example")(selection=f"default {operator} {SELECTION}")
            )

    def test_non_string_selection_raises_error(self, mock_schema):
        spy = SpySource(RawBand())
        with pytest.raises(exception.IncorrectUsage):
            list(DataAccess(spy, "example")(selection=123))

    def test_operations_from_data_pass_through(self):
        """from_data does not do schema lookup, so operations are just passed as-is."""
        raw = RawBand()
        for _, ctx in DataAccess.from_data(raw)(selection="A + B"):
            assert ctx.remaining_selection == "A + B"
