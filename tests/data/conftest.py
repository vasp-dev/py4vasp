from numpy.testing import assert_array_almost_equal_nulp
from contextlib import contextmanager
from unittest.mock import patch
import pytest
import py4vasp.raw as raw


class _Assert:
    @staticmethod
    def allclose(actual, desired):
        assert_array_almost_equal_nulp(actual, desired, 10)


@pytest.fixture
def Assert():
    return _Assert


@pytest.fixture
def mock_file():
    @contextmanager
    def _mock_file(name, ref):
        cm_init = patch.object(raw.File, "__init__", autospec=True, return_value=None)
        cm_sut = patch.object(raw.File, name, autospec=True, return_value=ref)
        cm_close = patch.object(raw.File, "close", autospec=True)
        with cm_init as init, cm_sut as sut, cm_close as close:
            yield {"init": init, "sut": sut, "close": close}

    return _mock_file


@pytest.fixture
def check_read():
    def _check_read(cls, mocks, ref, default_filename=None):
        ref = cls(ref)
        _check_read_from_open_file(cls, mocks, ref)
        _check_read_from_default_file(cls, mocks, ref, default_filename)
        _check_read_from_filename(cls, mocks, ref)

    def _check_read_from_open_file(cls, mocks, ref):
        with raw.File() as file:
            _reset_mocks(mocks)
            with cls.from_file(file) as actual:
                assert actual._raw == ref._raw
            mocks["init"].assert_not_called()
            mocks["sut"].assert_called_once()
            mocks["close"].assert_not_called()

    def _check_read_from_default_file(cls, mocks, ref, default_filename):
        _reset_mocks(mocks)
        with cls.from_file() as actual:
            assert actual._raw == ref._raw
        mocks["init"].assert_called_once()
        mocks["sut"].assert_called_once()
        mocks["close"].assert_called_once()
        args, _ = mocks["init"].call_args
        assert args[1] == default_filename

    def _check_read_from_filename(cls, mocks, ref):
        _reset_mocks(mocks)
        filename = "user_selected_file"
        with cls.from_file(filename) as actual:
            assert actual._raw == ref._raw
        mocks["init"].assert_called_once()
        mocks["sut"].assert_called_once()
        mocks["close"].assert_called_once()
        args, _ = mocks["init"].call_args
        assert args[1] == filename

    def _reset_mocks(mocks):
        for mock in mocks.values():
            mock.reset_mock()

    return _check_read
