# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pathlib
import re

import click

import py4vasp
from py4vasp import exception
from py4vasp._calculation.structure import Structure
from py4vasp._calculation.symmetry import _SYMPREC
from py4vasp._util import archive

_HDF5_SUFFIXES = (".h5", ".hdf5")


@click.group()
def cli():
    pass


@cli.command()
@click.argument("quantity", type=click.Choice(("structure",)), metavar="QUANTITY")
@click.argument("format", type=click.STRING)
@click.option(
    "-f",
    "--from",
    "path",
    type=click.Path(exists=True, readable=True),
    help="""Overwrite the default path where py4vasp looks for the quantity. This may
    be a directory, a file, or an archive of a VASP calculation.""",
)
@click.option(
    "-a",
    "--archive-path",
    "archive_path",
    type=click.STRING,
    help="Directory inside the archive in which the calculation is stored.",
)
@click.option(
    "-s",
    "--selection",
    type=click.STRING,
    help="String to further clarify the specific source of the quantity.",
)
def convert(quantity, format, path, archive_path, selection):
    """Convert a quantity to a different format.

    Specify which QUANTITY you want to convert into which FORMAT.
    """
    if format.lower() != "lammps":
        raise click.UsageError(f"Converting {quantity} to {format} is not implemented.")
    path = pathlib.Path.cwd() if path is None else pathlib.Path(path)
    try:
        calculation = _open_calculation(path, archive_path)
        result = _convert_to_lammps(calculation, selection)
    except exception.Py4VaspError as error:
        raise click.ClickException(*error.args) from error
    print(result)


def _open_calculation(path, archive_path):
    if archive.is_archive(path):
        return py4vasp.Calculation.from_archive(path, path=archive_path)
    if archive_path is not None:
        message = f"""\
The option --archive-path selects a calculation inside an archive, but {path} is not an
archive that py4vasp can read."""
        raise click.UsageError(message)
    if path.is_file():
        return py4vasp.Calculation.from_file(path)
    return py4vasp.Calculation.from_path(path)


def _convert_to_lammps(calculation, selection):
    if selection is None:
        result = calculation.structure.to_lammps()
    else:
        result = calculation.structure.to_lammps(selection=selection)
    return result


@cli.command()
@click.argument(
    "file", type=click.Path(exists=True, readable=True, path_type=pathlib.Path)
)
@click.option(
    "-p",
    "--primitive",
    is_flag=True,
    help="Reduce the structure to its primitive cell instead of keeping the input cell.",
)
@click.option(
    "--symprec",
    type=float,
    default=_SYMPREC,
    show_default=True,
    help="Symmetry tolerance in Å passed to spglib.",
)
@click.option(
    "-i",
    "--in-place",
    "in_place",
    is_flag=True,
    help="Overwrite FILE with the symmetrized structure.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=pathlib.Path),
    help="Write the symmetrized structure to this file instead of stdout.",
)
def symmetrize(file, primitive, symprec, in_place, output):
    """Symmetrize the structure in FILE and write it in POSCAR format.

    FILE may be a POSCAR, CONTCAR, or HDF5 file containing a structure, or an archive
    of a VASP calculation. By default
    the symmetrized structure is written in POSCAR format to stdout; use
    -o/--output or -i/--in-place to write it to a file instead.
    """
    if in_place and output:
        message = "The options -i/--in-place and -o/--output are mutually exclusive."
        raise click.UsageError(message)
    destination = file if in_place else output
    try:
        _raise_if_output_not_supported(destination, "the symmetrized structure")
        structure = _read_structure(file)
        result = structure.symmetrize(to_primitive=primitive, symprec=symprec)
        poscar = result.to_POSCAR()
    except exception.Py4VaspError as error:
        raise click.ClickException(*error.args) from error
    _write_or_print(poscar, destination)


def _write_or_print(text, destination):
    "Store the generated file where the user asked for it; stdout is the default."
    if destination is None:
        print(text)
    else:
        destination.write_text(text)


def _read_structure(file):
    if archive.is_archive(file):
        return py4vasp.Calculation.from_archive(file).structure
    if file.suffix in _HDF5_SUFFIXES:
        return py4vasp.Calculation.from_file(file).structure
    return Structure.from_POSCAR(file.read_text())


def _raise_if_output_not_supported(destination, description):
    "py4vasp generates text files, so they must not overwrite the data of a calculation."
    if destination is None:
        return
    if destination.suffix in _HDF5_SUFFIXES:
        message = f"Writing {description} to an HDF5 file is not implemented."
        raise exception.NotImplemented(message)
    if archive.is_archive(destination):
        message = (
            f"Writing {description} to an archive is not implemented. Note that it "
            "would replace the whole archive by a single text file."
        )
        raise exception.NotImplemented(message)


class _Divisions(click.ParamType):
    """The number of k points along the three directions as a single token.

    Passing the three numbers as separate arguments would make a variadic option
    swallow the file argument, so "8,8,6" is one token. A single number requests the
    same number of divisions along every direction.
    """

    name = "divisions"

    def convert(self, value, param, ctx):
        numbers = [number.strip() for number in str(value).split(",")]
        if not all(number.isdecimal() for number in numbers):
            self.fail(
                f"{value!r} is not a comma separated list of integers", param, ctx
            )
        numbers = [int(number) for number in numbers]
        if len(numbers) == 1:
            return tuple(numbers * 3)
        if len(numbers) != 3:
            message = f"expected one or three numbers, but {value!r} has {len(numbers)}"
            self.fail(message, param, ctx)
        return tuple(numbers)


@cli.group()
def generate():
    """Generate an input file for a VASP calculation."""


@generate.command("kpath")
@click.argument(
    "file", type=click.Path(exists=True, readable=True, path_type=pathlib.Path)
)
@click.option(
    "-n",
    "--number-points",
    "number_points",
    type=int,
    default=40,
    show_default=True,
    help="Number of k points VASP generates along every line of the path.",
)
@click.option(
    "--time-reversal/--no-time-reversal",
    "time_reversal",
    default=True,
    show_default=True,
    help="""Whether the band structure at k and -k agrees. Switch it off for a magnetic
    system without inversion symmetry; then the path also covers the primed images of
    the special points.""",
)
@click.option(
    "--symprec",
    type=float,
    default=_SYMPREC,
    show_default=True,
    help="Symmetry tolerance in Å passed to spglib.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=pathlib.Path),
    help="Write the KPOINTS file to this path instead of stdout.",
)
def generate_kpath(file, number_points, time_reversal, symprec, output):
    """Generate a KPOINTS file along the high-symmetry path of the structure in FILE.

    FILE may be a POSCAR, CONTCAR, or HDF5 file containing a structure, or an archive
    of a VASP calculation. seekpath determines the recommended path and py4vasp writes
    it in line mode with the label of every special point behind its coordinates, so
    that VASP reads the labels. By default the file is written to stdout; use
    -o/--output to store it as KPOINTS or KPOINTS_OPT.
    """
    try:
        _raise_if_output_not_supported(output, "a KPOINTS file")
        structure = _read_structure(file)
        kpoints = structure.generate_kpath(
            number_points=number_points,
            time_reversal=time_reversal,
            symprec=symprec,
        )
    except exception.Py4VaspError as error:
        raise click.ClickException(*error.args) from error
    _write_or_print(kpoints, output)


@generate.command("kmesh")
@click.argument(
    "file", type=click.Path(exists=True, readable=True, path_type=pathlib.Path)
)
@click.option(
    "--kspacing",
    type=float,
    help="""Largest allowed distance between two k points in Å⁻¹, following the
    convention of VASP's KSPACING tag. Specify either this or the divisions.""",
)
@click.option(
    "-d",
    "--divisions",
    type=_Divisions(),
    metavar="N,N,N",
    help="""Number of k points along the three directions of the conventional cell,
    e.g. 8,8,6 or 8 for the same number along every direction. Specify either this or
    the k-point spacing.""",
)
@click.option(
    "--shift",
    type=float,
    nargs=3,
    help="Shift of the mesh as fractions of its basis vectors.",
)
@click.option(
    "--symprec",
    type=float,
    default=_SYMPREC,
    show_default=True,
    help="Symmetry tolerance in Å passed to spglib.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=pathlib.Path),
    help="Write the KPOINTS file to this path instead of stdout.",
)
def generate_kmesh(file, kspacing, divisions, shift, symprec, output):
    """Generate a KPOINTS file with a mesh adapted to the structure in FILE.

    FILE may be a POSCAR, CONTCAR, or HDF5 file containing a structure, or an archive
    of a VASP calculation. The mesh subdivides the reciprocal lattice vectors of the
    conventional cell, so it retains the full symmetry of the lattice even when the
    calculation runs in the primitive cell. By default the file is written to stdout;
    use -o/--output to store it as KPOINTS.
    """
    divisions = divisions or None
    shift = shift or None
    if kspacing is None and divisions is None:
        message = (
            "Please specify how dense the mesh should be, either with --kspacing or "
            "with --divisions."
        )
        raise click.UsageError(message)
    if kspacing is not None and divisions is not None:
        message = (
            "--kspacing and --divisions both set the density of the mesh, so please "
            "specify only one of them."
        )
        raise click.UsageError(message)
    try:
        _raise_if_output_not_supported(output, "a KPOINTS file")
        structure = _read_structure(file)
        kpoints = structure.generate_kmesh(
            kspacing=kspacing,
            divisions=divisions,
            shift=shift,
            symprec=symprec,
        )
        lattice_vectors = structure.conventional_lattice_vectors(symprec=symprec)
    except exception.Py4VaspError as error:
        raise click.ClickException(*error.args) from error
    click.echo(_conventional_cell_report(kpoints, lattice_vectors), err=True)
    _write_or_print(kpoints, output)


def _conventional_cell_report(kpoints, lattice_vectors):
    """Tell the user which axis every division of the mesh belongs to.

    The mesh subdivides the conventional cell, which is not necessarily the cell in
    the input file, so without this the numbers passed to --divisions are a guess.
    It goes to stderr to keep the KPOINTS file on stdout byte for byte.
    """
    divisions = re.search(r"divisions ([\d ]+)", kpoints.splitlines()[0])
    subject = (
        f"Divisions {divisions.group(1).strip()} count"
        if divisions
        else "The divisions count"
    )
    rows = (
        f"  {name} " + " ".join(f"{component:12.8f}" for component in vector)
        for name, vector in zip("abc", lattice_vectors)
    )
    return "\n".join((f"{subject} along the conventional cell:", *rows))
