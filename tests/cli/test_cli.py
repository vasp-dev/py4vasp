# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import importlib
import pathlib
import subprocess
import sys
import zipfile
from unittest.mock import patch

import pytest

# The command line interface is shipped by the py4vasp distribution, not by
# py4vasp-core, so click is absent from a core-only installation.
click_testing = pytest.importorskip("click.testing")
CliRunner = click_testing.CliRunner

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


# ---------------------------------------------------------------------------
# module execution
# ---------------------------------------------------------------------------


def test_module_is_executable():
    """`python -m py4vasp` is what py4vasp-core[cli] users get; the console script
    itself is declared by the py4vasp distribution."""
    process = subprocess.run(
        [sys.executable, "-m", "py4vasp", "--help"], capture_output=True, text=True
    )
    assert process.returncode == 0, process.stderr
    assert "Usage" in process.stdout


def test_importing_the_module_does_not_run_the_command():
    """Without the __main__ guard, a package walk would exit the interpreter."""
    module = importlib.import_module("py4vasp.__main__")
    assert module.cli is cli


# ---------------------------------------------------------------------------
# archives
# ---------------------------------------------------------------------------


@pytest.fixture
def example_archive(tmp_path):
    filename = tmp_path / "calculation.zip"
    with zipfile.ZipFile(filename, "w") as zip_file:
        zip_file.writestr("relax/vaspout.h5", "not really HDF5")
    return filename


@pytest.mark.parametrize("argument", ("-f", "--from"))
def test_convert_archive(mock_calculation, example_archive, argument):
    runner = CliRunner()
    result = runner.invoke(
        cli, ["convert", "structure", "lammps", argument, str(example_archive)]
    )
    assert result.exit_code == 0
    mock_calculation.from_archive.assert_called_once_with(example_archive, path=None)
    structure = mock_calculation.from_archive.return_value.structure
    structure.to_lammps.assert_called_once_with()


@pytest.mark.parametrize("argument", ("-a", "--archive-path"))
def test_convert_archive_path(mock_calculation, example_archive, argument):
    runner = CliRunner()
    options = ["--from", str(example_archive), argument, "relax"]
    result = runner.invoke(cli, ["convert", "structure", "lammps", *options])
    assert result.exit_code == 0
    mock_calculation.from_archive.assert_called_once_with(example_archive, path="relax")


def test_convert_archive_path_without_archive(mock_calculation, tmp_path):
    runner = CliRunner()
    options = ["--from", str(tmp_path), "--archive-path", "relax"]
    result = runner.invoke(cli, ["convert", "structure", "lammps", *options])
    assert result.exit_code != 0
    assert "is not an\narchive that py4vasp can read" in result.output
    mock_calculation.from_path.assert_not_called()
    mock_calculation.from_archive.assert_not_called()


def test_symmetrize_archive(mock_calculation, example_archive):
    runner = CliRunner()
    result = runner.invoke(cli, ["symmetrize", str(example_archive)])
    assert result.exit_code == 0
    mock_calculation.from_archive.assert_called_once_with(example_archive)
    structure = mock_calculation.from_archive.return_value.structure
    structure.symmetrize.assert_called_once_with(to_primitive=False, symprec=_SYMPREC)
    symmetrized = structure.symmetrize.return_value
    assert result.output == f"{symmetrized.to_POSCAR.return_value}\n"


@pytest.mark.parametrize("flag", ("-i", "--in-place"))
def test_symmetrize_in_place_archive_not_implemented(
    mock_calculation, example_archive, flag
):
    original = example_archive.read_bytes()
    runner = CliRunner()
    result = runner.invoke(cli, ["symmetrize", str(example_archive), flag])
    assert result.exit_code != 0
    assert "archive" in result.output
    assert example_archive.read_bytes() == original
    mock_calculation.from_archive.assert_not_called()


def test_symmetrize_output_to_archive_not_implemented(
    mock_calculation, example_archive, tmp_path
):
    poscar = _write(tmp_path / "POSCAR")
    runner = CliRunner()
    options = ["--output", str(example_archive)]
    result = runner.invoke(cli, ["symmetrize", str(poscar), *options])
    assert result.exit_code != 0
    assert "archive" in result.output


# ---------------------------------------------------------------------------
# generate kpath command
# ---------------------------------------------------------------------------


def test_generate_kpath_writes_to_stdout(mock_structure, tmp_path):
    poscar = _write(tmp_path / "POSCAR")
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "kpath", str(poscar)])
    assert result.exit_code == 0
    mock_structure.from_POSCAR.assert_called_once_with("contents")
    structure = mock_structure.from_POSCAR.return_value
    structure.generate_kpath.assert_called_once_with(
        number_points=40, time_reversal=True, symprec=_SYMPREC
    )
    assert result.output == f"{structure.generate_kpath.return_value}\n"


@pytest.mark.parametrize("flag", ("-o", "--output"))
def test_generate_kpath_output_file(mock_structure, tmp_path, flag):
    poscar = _write(tmp_path / "POSCAR")
    output = tmp_path / "KPOINTS_OPT"
    mock_structure.from_POSCAR.return_value.generate_kpath.return_value = "KPOINTS"
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "kpath", str(poscar), flag, str(output)])
    assert result.exit_code == 0
    assert output.read_text() == "KPOINTS"
    assert result.output == ""  # nothing written to stdout


@pytest.mark.parametrize("flag", ("-n", "--number-points"))
def test_generate_kpath_forwards_number_points(mock_structure, tmp_path, flag):
    poscar = _write(tmp_path / "POSCAR")
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "kpath", str(poscar), flag, "20"])
    assert result.exit_code == 0
    structure = mock_structure.from_POSCAR.return_value
    structure.generate_kpath.assert_called_once_with(
        number_points=20, time_reversal=True, symprec=_SYMPREC
    )


def test_generate_kpath_forwards_symmetry_options(mock_structure, tmp_path):
    poscar = _write(tmp_path / "POSCAR")
    runner = CliRunner()
    options = ["--no-time-reversal", "--symprec", "0.1"]
    result = runner.invoke(cli, ["generate", "kpath", str(poscar), *options])
    assert result.exit_code == 0
    structure = mock_structure.from_POSCAR.return_value
    structure.generate_kpath.assert_called_once_with(
        number_points=40, time_reversal=False, symprec=0.1
    )


def test_generate_kpath_reports_py4vasp_error(mock_structure, tmp_path):
    poscar = _write(tmp_path / "POSCAR")
    output = tmp_path / "KPOINTS_OPT"
    error_message = "Cannot determine the path of a supercell."
    structure = mock_structure.from_POSCAR.return_value
    structure.generate_kpath.side_effect = exception.Py4VaspError(error_message)
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "kpath", str(poscar), "-o", str(output)])
    assert result.exit_code != 0
    assert error_message in result.output
    assert not output.exists()


def test_generate_kpath_missing_file():
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "kpath", "does_not_exist"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# generate kmesh command
# ---------------------------------------------------------------------------


def test_generate_kmesh_writes_to_stdout(mock_structure, tmp_path):
    poscar = _write(tmp_path / "POSCAR")
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "kmesh", str(poscar), "--kspacing", "0.2"])
    assert result.exit_code == 0
    mock_structure.from_POSCAR.assert_called_once_with("contents")
    structure = mock_structure.from_POSCAR.return_value
    structure.generate_kmesh.assert_called_once_with(
        kspacing=0.2, divisions=None, shift=None, symprec=_SYMPREC
    )
    assert result.output == f"{structure.generate_kmesh.return_value}\n"


@pytest.mark.parametrize("flag", ("-o", "--output"))
def test_generate_kmesh_output_file(mock_structure, tmp_path, flag):
    poscar = _write(tmp_path / "POSCAR")
    output = tmp_path / "KPOINTS"
    mock_structure.from_POSCAR.return_value.generate_kmesh.return_value = "MESH"
    runner = CliRunner()
    options = ["--kspacing", "0.2", flag, str(output)]
    result = runner.invoke(cli, ["generate", "kmesh", str(poscar), *options])
    assert result.exit_code == 0
    assert output.read_text() == "MESH"
    assert result.output == ""  # nothing written to stdout


@pytest.mark.parametrize("flag", ("-d", "--divisions"))
def test_generate_kmesh_forwards_divisions(mock_structure, tmp_path, flag):
    poscar = _write(tmp_path / "POSCAR")
    runner = CliRunner()
    options = [flag, "8", "8", "6", "--symprec", "0.1"]
    result = runner.invoke(cli, ["generate", "kmesh", str(poscar), *options])
    assert result.exit_code == 0
    structure = mock_structure.from_POSCAR.return_value
    structure.generate_kmesh.assert_called_once_with(
        kspacing=None, divisions=(8, 8, 6), shift=None, symprec=0.1
    )


def test_generate_kmesh_forwards_shift(mock_structure, tmp_path):
    poscar = _write(tmp_path / "POSCAR")
    runner = CliRunner()
    options = ["--kspacing", "0.2", "--shift", "0.5", "0.5", "0.0"]
    result = runner.invoke(cli, ["generate", "kmesh", str(poscar), *options])
    assert result.exit_code == 0
    structure = mock_structure.from_POSCAR.return_value
    structure.generate_kmesh.assert_called_once_with(
        kspacing=0.2, divisions=None, shift=(0.5, 0.5, 0.0), symprec=_SYMPREC
    )


@pytest.mark.parametrize(
    "options", ([], ["--kspacing", "0.2", "--divisions", "8", "8", "8"])
)
def test_generate_kmesh_without_density_fails(mock_structure, tmp_path, options):
    poscar = _write(tmp_path / "POSCAR")
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "kmesh", str(poscar), *options])
    assert result.exit_code != 0
    assert "--kspacing" in result.output and "--divisions" in result.output
    mock_structure.from_POSCAR.assert_not_called()


def test_generate_kmesh_reports_py4vasp_error(mock_structure, tmp_path):
    poscar = _write(tmp_path / "POSCAR")
    output = tmp_path / "KPOINTS"
    error_message = "Cannot determine the mesh of a supercell."
    structure = mock_structure.from_POSCAR.return_value
    structure.generate_kmesh.side_effect = exception.Py4VaspError(error_message)
    runner = CliRunner()
    options = ["--kspacing", "0.2", "-o", str(output)]
    result = runner.invoke(cli, ["generate", "kmesh", str(poscar), *options])
    assert result.exit_code != 0
    assert error_message in result.output
    assert not output.exists()
