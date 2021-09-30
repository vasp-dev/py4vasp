import pytest
from py4vasp.data import System
from py4vasp.raw import RawSystem
from py4vasp._util.convert import text_to_string


@pytest.fixture
def string_format():
    return RawSystem("string format")


@pytest.fixture
def byte_format():
    return RawSystem(b"byte format")


def test_system_print(string_format, byte_format, format_):
    check_system_print(string_format, format_)
    check_system_print(byte_format, format_)


def check_system_print(raw_system, format_):
    system = System(raw_system)
    actual, _ = format_(system)
    assert actual["text/plain"] == text_to_string(raw_system.system)


def test_descriptor(string_format, check_descriptors):
    descriptors = {"_to_string": ["__str__"]}
    check_descriptors(System(string_format), descriptors)


def test_from_file(string_format, mock_file, check_read):
    with mock_file("system", string_format) as mocks:
        check_read(System, mocks, string_format)
