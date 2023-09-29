# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

from py4vasp import data


@pytest.fixture
def workfunction(raw_data):
    raw_workfunction = raw_data.workfunction("default")
    return setup_reference(raw_workfunction)


def setup_reference(raw_workfunction):
    workfunction = data.Workfunction.from_data(raw_workfunction)
    return workfunction


def test_read(workfunction):
    assert True
