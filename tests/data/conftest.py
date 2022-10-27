# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from contextlib import contextmanager
from IPython.core.formatters import DisplayFormatter
from unittest.mock import patch, MagicMock, PropertyMock
import inspect
from pathlib import Path
import pytest
from py4vasp._util import convert
import py4vasp._util.version as version
from py4vasp import raw
from py4vasp import exceptions as exception


TEST_FILENAME = "read_data_from_this_file"


@pytest.fixture
def check_factory_methods():
    def inner(cls, data):
        instance = cls.from_path()
        check_instance_accesses_data(instance, data)
        instance = cls.from_file(TEST_FILENAME)
        check_instance_accesses_data(instance, data, file=TEST_FILENAME)

    return inner


def check_instance_accesses_data(instance, data, file=None):
    failed = []
    for name, method in inspect.getmembers(instance, inspect.ismethod):
        if should_test_method(name):
            try:
                check_method_accesses_data(data, method, file)
            except (AttributeError, AssertionError):
                failed.append(name)
    if failed:
        message = (
            f"The method(s) {', '.join(failed)} do not load the data from file."
            " The most likely issue is a missing @_base.data_access decorator."
        )
        raise AssertionError(message)


def should_test_method(name):
    if name in ("__str__", "_repr_html_"):
        return True
    if name.startswith("from") or name.startswith("_"):
        return False
    if name == "to_image":  # would have side effects
        return False
    return True


def check_method_accesses_data(data, method_under_test, file):
    quantity = convert.to_snakecase(data.__class__.__name__)
    with patch("py4vasp.raw.access") as mock_access:
        mock_access.return_value.__enter__.side_effect = lambda *_: data
        execute_method(method_under_test)
        check_mock_called(mock_access, quantity, file)
        mock_access.reset_mock()
        execute_method(method_under_test, source="choice")
        check_mock_called(mock_access, quantity, file, source="choice")


def execute_method(method_under_test, **kwargs):
    try:
        method_under_test(**kwargs)
    except (exception.NotImplemented, exception.IncorrectUsage):
        # ignore py4vasp error
        pass


def check_mock_called(mock_access, quantity, file, source=None):
    mock_access.assert_called_once()
    args, kwargs = mock_access.call_args
    assert (quantity,) == args
    assert kwargs.get("source") == source
    assert kwargs.get("file") == file


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
            file._path = Path.cwd()
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
        classname = f"{instance.__module__}.{instance.__class__.__name__}"
        for private_name, public_names in descriptors.items():
            fullname = f"{classname}.{private_name}"
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
