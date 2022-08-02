# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from unittest.mock import patch, mock_open
from pathlib import Path
import pytest
import py4vasp.data
import py4vasp.control as ctrl
import py4vasp.exceptions as exception
import inspect


@patch("py4vasp.data._base.Refinery.from_path", autospec=True)
@patch("py4vasp.raw.access", autospec=True)
def test_creation_from_path(mock_access, mock_from_path):
    # note: in pytest __file__ defaults to absolute path
    absolute_path = Path(__file__)
    calc = py4vasp.Calculation.from_path(absolute_path)
    assert calc.path() == absolute_path
    relative_path = absolute_path.relative_to(Path.cwd())
    calc = py4vasp.Calculation.from_path(relative_path)
    assert calc.path() == absolute_path
    calc = py4vasp.Calculation.from_path("~")
    assert calc.path() == Path.home()
    mock_access.assert_not_called()
    mock_from_path.assert_called()


@patch("py4vasp.data._base.Refinery.from_file", autospec=True)
@patch("py4vasp.raw.access", autospec=True)
def test_creation_from_file(mock_access, mock_from_file):
    # note: in pytest __file__ defaults to absolute path
    absolute_path = Path(__file__)
    absolute_file = absolute_path / "example.h5"
    calc = py4vasp.Calculation.from_file(absolute_file)
    assert calc.path() == absolute_path
    relative_file = absolute_file.relative_to(Path.cwd())
    calc = py4vasp.Calculation.from_file(relative_file)
    assert calc.path() == absolute_path
    calc = py4vasp.Calculation.from_file("~/example.h5")
    assert calc.path() == Path.home()
    mock_access.assert_not_called()
    mock_from_file.assert_called()


@patch("py4vasp.raw.access", autospec=True)
def test_all_attributes(mock_access):
    calculation = py4vasp.Calculation.from_path("test_path")
    camel_cases = {
        "BornEffectiveCharge": "born_effective_charge",
        "DielectricFunction": "dielectric_function",
        "DielectricTensor": "dielectric_tensor",
        "ElasticModulus": "elastic_modulus",
        "ForceConstant": "force_constant",
        "InternalStrain": "internal_strain",
        "PairCorrelation": "pair_correlation",
        "PhononBand": "phonon_band",
        "PhononDos": "phonon_dos",
        "PiezoelectricTensor": "piezoelectric_tensor",
    }
    for name, _ in inspect.getmembers(py4vasp.data, inspect.isclass):
        attr = camel_cases.get(name, name.lower())
        assert hasattr(calculation, attr)
    mock_access.assert_not_called()
    mock_access.return_value.__enter__.assert_not_called()


def test_input_files_from_path():
    with patch("py4vasp.control._base.InputBase.__init__", return_value=None) as mock:
        calculation = py4vasp.Calculation.from_path("test_path")
        mock.assert_called_with(calculation.path())
    calculation = py4vasp.Calculation.from_path("test_path")
    check_all_input_files(calculation)


def test_input_files_from_file():
    with patch("py4vasp.control._base.InputBase.__init__", return_value=None) as mock:
        calculation = py4vasp.Calculation.from_file("test_file")
        mock.assert_called_with(calculation.path())
    calculation = py4vasp.Calculation.from_file("test_file")
    check_all_input_files(calculation)


def check_all_input_files(calculation):
    input_files = [ctrl.INCAR, ctrl.KPOINTS, ctrl.POSCAR]
    for input_file in input_files:
        check_one_input_file(calculation, input_file)


def check_one_input_file(calculation, input_file):
    text = "! comment line"
    name = input_file.__name__
    assert isinstance(getattr(calculation, name), input_file)
    with patch("py4vasp.control._base.open", mock_open(read_data=text)) as mock:
        setattr(calculation, name, text)
        mock.assert_called_once_with(calculation.path() / name, "w")
        mock.reset_mock()
        assert getattr(calculation, name).read() == text
        mock.assert_called_once_with(calculation.path() / name, "r")


def test_using_constructor_raises_exception():
    with pytest.raises(exception.IncorrectUsage):
        py4vasp.Calculation()
    with pytest.raises(exception.IncorrectUsage):
        py4vasp.Calculation("path")
    with pytest.raises(exception.IncorrectUsage):
        py4vasp.Calculation(key="value")
