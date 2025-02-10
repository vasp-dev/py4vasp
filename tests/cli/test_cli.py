# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pathlib
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from py4vasp.cli import cli


@pytest.fixture
def mock_calculation():
    with patch("py4vasp.Calculation", autospec=True) as mock:
        yield mock


@pytest.mark.parametrize("lammps", ("LAMMPS", "Lammps", "lammps"))
def test_convert_lammps(mock_calculation, lammps):
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "structure", lammps])
    assert result.exit_code == 0
    expected_path = pathlib.Path.cwd()
    constructor = mock_calculation.from_path
    constructor.assert_called_once_with(expected_path)
    structure = constructor.return_value.structure
    structure.to_lammps.assert_called_once_with()
    converted = structure.to_lammps.return_value
    assert f"{converted}\n" == result.output


def test_convert_wrong_quantity():
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "not_implemented"])
    assert result.exit_code != 0
    assert "Invalid value" in result.output
    assert "not_implemented" in result.output


def test_convert_wrong_format(mock_calculation):
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "structure", "not_implemented"])
    assert result.exit_code != 0
    mock_calculation.from_path.assert_not_called()


@pytest.mark.parametrize("position", ("first", "middle", "last"))
@pytest.mark.parametrize("selection", (("-s", "choice"), ("--selection", "choice")))
def test_convert_selection(mock_calculation, position, selection):
    runner = CliRunner()
    result = invoke_runner_with_options(runner, position, selection)
    assert result.exit_code == 0
    expected_path = pathlib.Path.cwd()
    constructor = mock_calculation.from_path
    constructor.assert_called_once_with(expected_path)
    structure = constructor.return_value.structure
    structure.to_lammps.assert_called_once_with(selection="choice")
    converted = structure.to_lammps.return_value
    assert f"{converted}\n" == result.output


@pytest.mark.parametrize("position", ("first", "middle", "last"))
@pytest.mark.parametrize("argument", ("-f", "--from"))
@pytest.mark.parametrize("path", ("dirname", "filename"))
def test_convert_path(mock_calculation, position, argument, path, tmp_path):
    expected_path = tmp_path / path
    if path == "dirname":
        expected_path.mkdir()
    else:
        expected_path.touch()
    runner = CliRunner()
    result = invoke_runner_with_options(runner, position, (argument, expected_path))
    print(result.output)
    assert result.exit_code == 0
    if path == "filename":
        constructor = mock_calculation.from_file
    else:
        constructor = mock_calculation.from_path
    constructor.assert_called_once_with(expected_path)
    structure = constructor.return_value.structure
    structure.to_lammps.assert_called_once_with()
    converted = structure.to_lammps.return_value
    assert f"{converted}\n" == result.output


def invoke_runner_with_options(runner, position, options):
    if position == "first":
        return runner.invoke(cli, ["convert", *options, "structure", "lammps"])
    elif position == "middle":
        return runner.invoke(cli, ["convert", "structure", *options, "lammps"])
    elif position == "last":
        return runner.invoke(cli, ["convert", "structure", "lammps", *options])
    else:
        raise NotImplementedError
