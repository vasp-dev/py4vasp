# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pathlib
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from py4vasp import exception
from py4vasp._calculation.symmetry import _SYMPREC
from py4vasp.cli import cli


@pytest.fixture
def mock_calculation():
    with patch("py4vasp.Calculation", autospec=True) as mock:
        yield mock


@pytest.fixture
def mock_structure():
    with patch("py4vasp.cli.Structure", autospec=True) as mock:
        yield mock


@pytest.mark.parametrize("lammps", ("LAMMPS", "Lammps", "lammps"))
def test_convert_lammps(mock_calculation, lammps):
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "structure", lammps])
    assert result.exit_code == 0
    check_conversion_called(mock_calculation, result)


@pytest.mark.parametrize("position", ("first", "middle", "last"))
@pytest.mark.parametrize("selection", (("-s", "choice"), ("--selection", "choice")))
def test_convert_selection(mock_calculation, position, selection):
    runner = CliRunner()
    result = invoke_runner_with_options(runner, position, selection)
    check_conversion_called(mock_calculation, result, selection=selection[1])


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
    check_conversion_called(mock_calculation, result, expected_path=expected_path)


def invoke_runner_with_options(runner, position, options):
    if position == "first":
        return runner.invoke(cli, ["convert", *options, "structure", "lammps"])
    elif position == "middle":
        return runner.invoke(cli, ["convert", "structure", *options, "lammps"])
    elif position == "last":
        return runner.invoke(cli, ["convert", "structure", "lammps", *options])
    else:
        raise NotImplementedError


def check_conversion_called(
    mock_calculation, result, selection=None, expected_path=pathlib.Path.cwd()
):
    assert result.exit_code == 0
    if expected_path.name == "filename":
        constructor = mock_calculation.from_file
    else:
        constructor = mock_calculation.from_path
    constructor.assert_called_once_with(expected_path)
    structure = constructor.return_value.structure
    if selection is None:
        structure.to_lammps.assert_called_once_with()
    else:
        structure.to_lammps.assert_called_once_with(selection=selection)
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


def test_error_in_py4vasp(mock_calculation):
    runner = CliRunner()
    error_message = "Custom error message."
    mock_calculation.from_path.side_effect = exception.Py4VaspError(error_message)
    result = runner.invoke(cli, ["convert", "structure", "lammps"])
    assert result.exit_code != 0
    assert error_message in result.output


# ---------------------------------------------------------------------------
# symmetrize command
# ---------------------------------------------------------------------------


def _write(path, text="contents"):
    path.write_text(text)
    return path


@pytest.mark.parametrize("filename", ("POSCAR", "CONTCAR", "structure.vasp"))
def test_symmetrize_poscar_to_stdout(mock_structure, tmp_path, filename):
    poscar = _write(tmp_path / filename)
    runner = CliRunner()
    result = runner.invoke(cli, ["symmetrize", str(poscar)])
    assert result.exit_code == 0
    mock_structure.from_POSCAR.assert_called_once_with("contents")
    structure = mock_structure.from_POSCAR.return_value
    structure.symmetrize.assert_called_once_with(to_primitive=False, symprec=_SYMPREC)
    symmetrized = structure.symmetrize.return_value
    symmetrized.to_POSCAR.assert_called_once_with()
    assert result.output == f"{symmetrized.to_POSCAR.return_value}\n"


@pytest.mark.parametrize("suffix", (".h5", ".hdf5"))
def test_symmetrize_hdf5_reads_via_calculation(mock_calculation, tmp_path, suffix):
    hdf5 = _write(tmp_path / f"vaspout{suffix}")
    runner = CliRunner()
    result = runner.invoke(cli, ["symmetrize", str(hdf5)])
    assert result.exit_code == 0
    mock_calculation.from_file.assert_called_once_with(hdf5)
    structure = mock_calculation.from_file.return_value.structure
    structure.symmetrize.assert_called_once_with(to_primitive=False, symprec=_SYMPREC)
    symmetrized = structure.symmetrize.return_value
    assert result.output == f"{symmetrized.to_POSCAR.return_value}\n"


@pytest.mark.parametrize("flag", ("-p", "--primitive"))
def test_symmetrize_primitive_flag(mock_structure, tmp_path, flag):
    poscar = _write(tmp_path / "POSCAR")
    runner = CliRunner()
    result = runner.invoke(cli, ["symmetrize", str(poscar), flag])
    assert result.exit_code == 0
    structure = mock_structure.from_POSCAR.return_value
    structure.symmetrize.assert_called_once_with(to_primitive=True, symprec=_SYMPREC)


def test_symmetrize_symprec_option(mock_structure, tmp_path):
    poscar = _write(tmp_path / "POSCAR")
    runner = CliRunner()
    result = runner.invoke(cli, ["symmetrize", str(poscar), "--symprec", "0.1"])
    assert result.exit_code == 0
    structure = mock_structure.from_POSCAR.return_value
    structure.symmetrize.assert_called_once_with(to_primitive=False, symprec=0.1)


def test_symmetrize_error_in_py4vasp(mock_structure, tmp_path):
    poscar = _write(tmp_path / "POSCAR")
    error_message = "Cannot symmetrize."
    mock_structure.from_POSCAR.side_effect = exception.Py4VaspError(error_message)
    runner = CliRunner()
    result = runner.invoke(cli, ["symmetrize", str(poscar)])
    assert result.exit_code != 0
    assert error_message in result.output


def test_symmetrize_missing_file():
    runner = CliRunner()
    result = runner.invoke(cli, ["symmetrize", "does_not_exist"])
    assert result.exit_code != 0


def _set_symmetrized_poscar(mock_structure, text):
    structure = mock_structure.from_POSCAR.return_value
    structure.symmetrize.return_value.to_POSCAR.return_value = text


@pytest.mark.parametrize("flag", ("-o", "--output"))
def test_symmetrize_output_file(mock_structure, tmp_path, flag):
    poscar = _write(tmp_path / "POSCAR")
    output = tmp_path / "symmetrized.vasp"
    _set_symmetrized_poscar(mock_structure, "SYMMETRIZED")
    runner = CliRunner()
    result = runner.invoke(cli, ["symmetrize", str(poscar), flag, str(output)])
    assert result.exit_code == 0
    assert output.read_text() == "SYMMETRIZED"
    assert result.output == ""  # nothing written to stdout


@pytest.mark.parametrize("flag", ("-i", "--in-place"))
def test_symmetrize_in_place(mock_structure, tmp_path, flag):
    poscar = _write(tmp_path / "POSCAR")
    _set_symmetrized_poscar(mock_structure, "SYMMETRIZED")
    runner = CliRunner()
    result = runner.invoke(cli, ["symmetrize", str(poscar), flag])
    assert result.exit_code == 0
    mock_structure.from_POSCAR.assert_called_once_with("contents")
    assert poscar.read_text() == "SYMMETRIZED"
    assert result.output == ""


def test_symmetrize_in_place_and_output_are_exclusive(mock_structure, tmp_path):
    poscar = _write(tmp_path / "POSCAR")
    output = tmp_path / "out.vasp"
    runner = CliRunner()
    result = runner.invoke(cli, ["symmetrize", str(poscar), "-i", "-o", str(output)])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


@pytest.mark.parametrize("suffix", (".h5", ".hdf5"))
def test_symmetrize_output_to_hdf5_not_implemented(mock_structure, tmp_path, suffix):
    poscar = _write(tmp_path / "POSCAR")
    output = tmp_path / f"out{suffix}"
    runner = CliRunner()
    result = runner.invoke(cli, ["symmetrize", str(poscar), "-o", str(output)])
    assert result.exit_code != 0
    assert "not implemented" in result.output.lower()
    assert not output.exists()


def test_symmetrize_in_place_hdf5_not_implemented(mock_calculation, tmp_path):
    hdf5 = _write(tmp_path / "vaspout.h5")
    runner = CliRunner()
    result = runner.invoke(cli, ["symmetrize", str(hdf5), "-i"])
    assert result.exit_code != 0
    assert "not implemented" in result.output.lower()
