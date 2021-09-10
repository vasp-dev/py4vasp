from numpy.testing import assert_array_almost_equal_nulp
from contextlib import contextmanager
from IPython.core.formatters import DisplayFormatter
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path
import pytest
import py4vasp._util.version as version
import py4vasp.raw as raw


class _Assert:
    @staticmethod
    def allclose(actual, desired):
        if actual is None:
            assert desired is None
        else:
            assert_array_almost_equal_nulp(actual, desired, 10)


@pytest.fixture
def Assert():
    return _Assert


@pytest.fixture
def mock_file():
    @contextmanager
    def _mock_file(name, ref):
        cm_init = patch.object(raw.File, "__init__", autospec=True, return_value=None)
        cm_sut = patch.object(raw.File, name, new_callable=PropertyMock)
        cm_close = patch.object(raw.File, "close", autospec=True)
        with cm_init as init, cm_sut as sut, cm_close as close:
            sut.return_value = {"default": ref}
            yield {"init": init, "sut": sut, "close": close}

    return _mock_file


@pytest.fixture
def check_read():
    def _check_read(cls, mocks, ref, default_filename=None):
        _check_read_from_open_file(cls, mocks, ref)
        _check_read_from_default_file(cls, mocks, ref, default_filename)
        _check_read_from_filename(cls, mocks, ref)
        _check_read_from_path(cls, mocks, ref, default_filename)

    def _check_read_from_open_file(cls, mocks, ref):
        with raw.File() as file:
            obj = _create_obj(cls, file, _assert_not_called, mocks)
            _check_raw_data(obj, ref, _assert_only_sut, mocks)

    def _check_read_from_default_file(cls, mocks, ref, default_filename):
        obj = _create_obj(cls, None, _assert_not_called, mocks)
        _check_raw_data(obj, ref, _assert_all_called, mocks)
        assert _first_init_arg(mocks) == default_filename

    def _check_read_from_filename(cls, mocks, ref):
        filename = "user_selected_file"
        obj = _create_obj(cls, filename, _assert_not_called, mocks)
        _check_raw_data(obj, ref, _assert_all_called, mocks)
        assert _first_init_arg(mocks) == filename

    def _check_read_from_path(cls, mocks, ref, default_filename):
        path = Path.cwd()
        obj = _create_obj(cls, path, _assert_not_called, mocks)
        _check_raw_data(obj, ref, _assert_all_called, mocks)
        if default_filename is None:
            assert _first_init_arg(mocks) == path
        else:
            assert _first_init_arg(mocks) == path / default_filename

    def _create_obj(cls, file, assertion, mocks):
        _reset_mocks(mocks)
        obj = cls.from_file(file)
        assertion(mocks)
        return obj

    def _check_raw_data(obj, ref, assertion, mocks):
        _reset_mocks(mocks)
        with obj._data_dict_from_context() as actual:
            assert actual["default"] == ref
        assertion(mocks)

    def _assert_not_called(mocks):
        mocks["init"].assert_not_called()
        mocks["sut"].assert_not_called()
        mocks["close"].assert_not_called()

    def _assert_only_sut(mocks):
        mocks["init"].assert_not_called()
        mocks["sut"].assert_called_once()
        mocks["close"].assert_not_called()

    def _assert_all_called(mocks):
        mocks["init"].assert_called_once()
        mocks["sut"].assert_called_once()
        mocks["close"].assert_called_once()

    def _first_init_arg(mocks):
        args, _ = mocks["init"].call_args
        return args[1]

    def _reset_mocks(mocks):
        for mock in mocks.values():
            mock.reset_mock()

    return _check_read


@pytest.fixture
def check_descriptors():
    def _check_descriptors(instance, descriptors):
        for private_name, public_names in descriptors.items():
            fullname = f"{instance.__module__}.{private_name}"
            with patch(fullname, return_value=private_name):
                for public_name in public_names:
                    assert private_name == getattr(instance, public_name)()

    return _check_descriptors


@pytest.fixture
def outdated_version():
    return raw.RawVersion(version.minimal_vasp_version.major - 1)


@pytest.fixture
def format_():
    return DisplayFormatter().format
