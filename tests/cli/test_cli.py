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
        structure = mock.from_POSCAR.return_value
        structure.conventional_lattice_vectors.return_value = _CONVENTIONAL_CELL
        structure.generate_kmesh.return_value = _KMESH_TEXT
        structure.generate_kpath.return_value = _KPATH_TEXT
        yield mock


# a slab, so that the three axes are told apart in the report
_CONVENTIONAL_CELL = [[3.16, 0.0, 0.0], [-1.58, 2.7366, 0.0], [0.0, 0.0, 20.0]]
# the command reads the divisions back out of the text it generated
_KMESH_TEXT = "k mesh of the conventional cell: divisions 4 4 4\n0\nReduced"
# the first line carries the space group the --symprec help tells the user to check
_KPATH_TEXT = (
    "k points along high symmetry lines: Fd-3m (cF2), 1 primitive cell per unit cell"
    "\n40\nline mode\nreciprocal"
)


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
    assert result.stdout == f"{structure.generate_kpath.return_value}\n"


@pytest.mark.parametrize("flag", ("-o", "--output"))
def test_generate_kpath_output_file(mock_structure, tmp_path, flag):
    poscar = _write(tmp_path / "POSCAR")
    output = tmp_path / "KPOINTS_OPT"
    mock_structure.from_POSCAR.return_value.generate_kpath.return_value = "KPOINTS"
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "kpath", str(poscar), flag, str(output)])
    assert result.exit_code == 0
    assert output.read_text() == "KPOINTS"
    assert result.stdout == ""  # the file is not echoed to stdout


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
    assert result.stdout == f"{structure.generate_kmesh.return_value}\n"


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
    assert result.stdout == ""  # the file is not echoed to stdout
    assert "conventional cell" in result.stderr  # but the report still reaches the user


@pytest.mark.parametrize("flag", ("-d", "--divisions"))
@pytest.mark.parametrize("options_first", (True, False))
def test_generate_kmesh_forwards_divisions(
    mock_structure, tmp_path, flag, options_first
):
    poscar = _write(tmp_path / "POSCAR")
    runner = CliRunner()
    options = [flag, "8,8,6", "--symprec", "0.1"]
    # a single token cannot swallow the file argument, whichever order they come in
    arguments = [*options, str(poscar)] if options_first else [str(poscar), *options]
    result = runner.invoke(cli, ["generate", "kmesh", *arguments])
    assert result.exit_code == 0
    structure = mock_structure.from_POSCAR.return_value
    structure.generate_kmesh.assert_called_once_with(
        kspacing=None, divisions=(8, 8, 6), shift=None, symprec=0.1
    )


def test_generate_kmesh_divisions_shorthand(mock_structure, tmp_path):
    poscar = _write(tmp_path / "POSCAR")
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "kmesh", str(poscar), "-d", "8"])
    assert result.exit_code == 0
    structure = mock_structure.from_POSCAR.return_value
    structure.generate_kmesh.assert_called_once_with(
        kspacing=None, divisions=(8, 8, 8), shift=None, symprec=_SYMPREC
    )


@pytest.mark.parametrize("value", ("8,8", "8,8,6,6", "eight", "8,8,six", ""))
def test_generate_kmesh_rejects_malformed_divisions(mock_structure, tmp_path, value):
    poscar = _write(tmp_path / "POSCAR")
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "kmesh", str(poscar), "-d", value])
    assert result.exit_code != 0
    assert "--divisions" in result.output
    mock_structure.from_POSCAR.assert_not_called()


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
    "options, expected",
    [
        ([], "not"),  # neither given: must not complain about both being given
        (["--kspacing", "0.2", "--divisions", "8,8,8"], "only one"),
    ],
)
def test_generate_kmesh_without_density_fails(
    mock_structure, tmp_path, options, expected
):
    poscar = _write(tmp_path / "POSCAR")
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "kmesh", str(poscar), *options])
    assert result.exit_code != 0
    assert "--kspacing" in result.output and "--divisions" in result.output
    if expected == "only one":
        assert "only one" in result.output
    else:
        assert "only one" not in result.output
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


# ---------------------------------------------------------------------------
# generate does not overwrite the input files of a calculation
# ---------------------------------------------------------------------------

_GENERATE_COMMANDS = [("kpath", []), ("kmesh", ["--kspacing", "0.2"])]


@pytest.mark.parametrize("command, options", _GENERATE_COMMANDS)
@pytest.mark.parametrize("suffix", (".h5", ".hdf5"))
def test_generate_does_not_overwrite_hdf5(
    mock_structure, tmp_path, command, options, suffix
):
    poscar = _write(tmp_path / "POSCAR")
    output = _write(tmp_path / f"vaspout{suffix}", "not really HDF5")
    runner = CliRunner()
    options = [*options, "-o", str(output)]
    result = runner.invoke(cli, ["generate", command, str(poscar), *options])
    assert result.exit_code != 0
    assert "not implemented" in result.output.lower()
    assert output.read_text() == "not really HDF5"  # the file is untouched
    mock_structure.from_POSCAR.assert_not_called()


@pytest.mark.parametrize("command, options", _GENERATE_COMMANDS)
def test_generate_does_not_overwrite_archive(
    mock_structure, example_archive, tmp_path, command, options
):
    poscar = _write(tmp_path / "POSCAR")
    original = example_archive.read_bytes()
    runner = CliRunner()
    options = [*options, "-o", str(example_archive)]
    result = runner.invoke(cli, ["generate", command, str(poscar), *options])
    assert result.exit_code != 0
    assert "archive" in result.output
    assert example_archive.read_bytes() == original
    mock_structure.from_POSCAR.assert_not_called()


def test_generate_kmesh_reports_the_conventional_cell(mock_structure, tmp_path):
    poscar = _write(tmp_path / "POSCAR")
    structure = mock_structure.from_POSCAR.return_value
    structure.generate_kmesh.return_value = (
        "k mesh of the conventional cell: divisions 2 10 6, kspacing 0.2\n0\nReduced"
    )
    runner = CliRunner()
    options = ["--kspacing", "0.2", "--symprec", "0.1"]
    result = runner.invoke(cli, ["generate", "kmesh", str(poscar), *options])
    assert result.exit_code == 0
    # the file itself stays a KPOINTS file, byte for byte
    assert result.stdout == f"{structure.generate_kmesh.return_value}\n"
    # the report tells the user which axis each division belongs to
    assert "Divisions 2 10 6" in result.stderr
    assert "conventional cell" in result.stderr
    for name, vector in zip("abc", _CONVENTIONAL_CELL):
        assert f"  {name} " in result.stderr
        assert f"{vector[0]:12.8f}" in result.stderr
    structure.conventional_lattice_vectors.assert_called_once_with(symprec=0.1)


def test_generate_kpath_does_not_report_a_conventional_cell(mock_structure, tmp_path):
    poscar = _write(tmp_path / "POSCAR")
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "kpath", str(poscar)])
    assert result.exit_code == 0
    # the path does not count divisions along the conventional cell, so it says nothing
    # about that cell -- but it does report the space group, see the test below
    assert "conventional cell" not in result.stderr


@pytest.mark.parametrize("writes_a_file", (True, False))
def test_generate_kpath_reports_the_space_group(
    mock_structure, tmp_path, writes_a_file
):
    poscar = _write(tmp_path / "POSCAR")
    output = tmp_path / "KPOINTS_OPT"
    options = ["-o", str(output)] if writes_a_file else []
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "kpath", str(poscar), *options])
    assert result.exit_code == 0
    # the --symprec help tells the user to check the space group, so it has to be
    # visible whether the file goes to stdout, to -o, or into a shell redirection
    assert "Fd-3m (cF2)" in result.stderr
    assert "1 primitive cell" in result.stderr


# ---------------------------------------------------------------------------
# generate reports mistakes without a traceback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command, options", _GENERATE_COMMANDS)
def test_generate_rejects_a_missing_output_directory(
    mock_structure, tmp_path, command, options
):
    poscar = _write(tmp_path / "POSCAR")
    output = tmp_path / "nosuchdir" / "KPOINTS"
    runner = CliRunner()
    options = [*options, "-o", str(output)]
    result = runner.invoke(cli, ["generate", command, str(poscar), *options])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "nosuchdir" in result.output
    assert not output.exists()


@pytest.mark.parametrize("command, options", _GENERATE_COMMANDS)
def test_generate_rejects_a_directory_as_input(
    mock_structure, tmp_path, command, options
):
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", command, str(tmp_path), *options])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "directory" in result.output.lower()
    mock_structure.from_POSCAR.assert_not_called()


@pytest.mark.parametrize("command, options", _GENERATE_COMMANDS)
def test_generate_forwards_elements(mock_structure, tmp_path, command, options):
    # old POSCAR files do not name their elements, and the user must be able to say
    # so from the command line rather than being told to call a Python routine
    poscar = _write(tmp_path / "POSCAR")
    runner = CliRunner()
    options = [*options, "--elements", "Si,O"]
    result = runner.invoke(cli, ["generate", command, str(poscar), *options])
    assert result.exit_code == 0
    mock_structure.from_POSCAR.assert_called_once_with("contents", elements=["Si", "O"])
