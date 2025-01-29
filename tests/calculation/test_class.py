# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import os
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from py4vasp import Calculation, _calculation, control, exception
from py4vasp._calculation import base


@patch.object(base.Refinery, "from_path", autospec=True)
@patch("py4vasp.raw.access", autospec=True)
def test_creation_from_path(mock_access, mock_from_path):
    # note: in pytest __file__ defaults to absolute path
    absolute_path = Path(__file__)
    calc = Calculation.from_path(absolute_path)
    assert calc.path() == absolute_path
    relative_path = os.path.relpath(absolute_path, Path.cwd())
    calc = Calculation.from_path(relative_path)
    assert calc.path() == absolute_path
    calc = Calculation.from_path("~")
    assert calc.path() == Path.home()
    mock_access.assert_not_called()
    mock_from_path.assert_not_called()
    calc.band  # access the band object
    mock_from_path.assert_called_once()


@patch.object(base.Refinery, "from_file", autospec=True)
@patch("py4vasp.raw.access", autospec=True)
def test_creation_from_file(mock_access, mock_from_file):
    # note: in pytest __file__ defaults to absolute path
    absolute_path = Path(__file__)
    absolute_file = absolute_path / "example.h5"
    calc = Calculation.from_file(absolute_file)
    assert calc.path() == absolute_path
    relative_file = os.path.relpath(absolute_file, Path.cwd())
    calc = Calculation.from_file(relative_file)
    assert calc.path() == absolute_path
    calc = Calculation.from_file("~/example.h5")
    assert calc.path() == Path.home()
    mock_access.assert_not_called()
    mock_from_file.assert_not_called()
    calc.band  # access the band object
    mock_from_file.assert_called()


@patch("py4vasp.raw.access", autospec=True)
def test_all_attributes(mock_access):
    calc = Calculation.from_path("test_path")
    for name in _calculation.QUANTITIES:  #  + _calculation.INPUT_FILES:
        assert hasattr(calc, name)
    mock_access.assert_not_called()
    mock_access.return_value.__enter__.assert_not_called()


@pytest.mark.skip("Input files are not included in current release.")
def test_input_files_from_path():
    with patch("py4vasp._control.base.InputFile.__init__", return_value=None) as mock:
        calculation = Calculation.from_path("test_path")
        mock.assert_called_with(calculation.path())
    calculation = Calculation.from_path("test_path")
    check_all_input_files(calculation)


@pytest.mark.skip("Input files are not included in current release.")
def test_input_files_from_file():
    with patch("py4vasp._control.base.InputFile.__init__", return_value=None) as mock:
        calculation = Calculation.from_file("test_file")
        mock.assert_called_with(calculation.path())
    calculation = Calculation.from_file("test_file")
    check_all_input_files(calculation)


def check_all_input_files(calculation):
    input_files = [control.INCAR, control.KPOINTS, control.POSCAR]
    for input_file in input_files:
        check_one_input_file(calculation, input_file)


def check_one_input_file(calculation, input_file):
    text = "! comment line"
    name = input_file.__name__
    assert isinstance(getattr(calculation, name), input_file)
    with patch("py4vasp._control.base.open", mock_open(read_data=text)) as mock:
        setattr(calculation, name, text)
        mock.assert_called_once_with(calculation.path() / name, "w")
        mock.reset_mock()
        assert getattr(calculation, name).read() == text
        mock.assert_called_once_with(calculation.path() / name, "r")


def test_using_constructor_raises_exception():
    with pytest.raises(exception.IncorrectUsage):
        Calculation()
    with pytest.raises(exception.IncorrectUsage):
        Calculation("path")
    with pytest.raises(exception.IncorrectUsage):
        Calculation(key="value")
