# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import string
import pytest
from py4vasp.data import System
from py4vasp import raw
from py4vasp._util.convert import text_to_string


@pytest.fixture
def string_format():
    return raw.System("string format")


@pytest.fixture
def byte_format():
    return raw.System(b"byte format")


def test_system_read(string_format, byte_format):
    check_system_read(string_format)
    check_system_read(byte_format)


def check_system_read(raw_system):
    expected = {"system": text_to_string(raw_system.system)}
    assert System.from_data(raw_system).read() == expected


def test_system_print(string_format, byte_format, format_):
    check_system_print(string_format, format_)
    check_system_print(byte_format, format_)


def check_system_print(raw_system, format_):
    system = System.from_data(raw_system)
    actual, _ = format_(system)
    assert actual["text/plain"] == text_to_string(raw_system.system)


def test_factory_methods(string_format, check_factory_methods):
    check_factory_methods(System, string_format)
