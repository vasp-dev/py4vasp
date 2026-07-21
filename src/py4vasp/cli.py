# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pathlib

import click

import py4vasp
from py4vasp import exception
from py4vasp._calculation.structure import Structure
from py4vasp._calculation.symmetry import _SYMPREC

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
    help="Overwrite the default path where py4vasp looks for the quantity.",
)
@click.option(
    "-s",
    "--selection",
    type=click.STRING,
    help="String to further clarify the specific source of the quantity.",
)
def convert(quantity, format, path, selection):
    """Convert a quantity to a different format.

    Specify which QUANTITY you want to convert into which FORMAT.
    """
    if format.lower() != "lammps":
        raise click.UsageError(f"Converting {quantity} to {format} is not implemented.")
    path = pathlib.Path.cwd() if path is None else pathlib.Path(path)
    try:
        result = _convert_to_lammps(path, selection)
    except exception.Py4VaspError as error:
        raise click.ClickException(*error.args) from error
    print(result)


def _convert_to_lammps(path, selection):
    if path.is_file():
        calculation = py4vasp.Calculation.from_file(path)
    else:
        calculation = py4vasp.Calculation.from_path(path)
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

    FILE may be a POSCAR, CONTCAR, or HDF5 file containing a structure. By default
    the symmetrized structure is written in POSCAR format to stdout; use
    -o/--output or -i/--in-place to write it to a file instead.
    """
    if in_place and output:
        message = "The options -i/--in-place and -o/--output are mutually exclusive."
        raise click.UsageError(message)
    destination = file if in_place else output
    try:
        _raise_if_hdf5_output(destination)
        structure = _read_structure(file)
        result = structure.symmetrize(to_primitive=primitive, symprec=symprec)
        poscar = result.to_POSCAR()
    except exception.Py4VaspError as error:
        raise click.ClickException(*error.args) from error
    if destination is None:
        print(poscar)
    else:
        destination.write_text(poscar)


def _read_structure(file):
    if file.suffix in _HDF5_SUFFIXES:
        return py4vasp.Calculation.from_file(file).structure
    return Structure.from_POSCAR(file.read_text())


def _raise_if_hdf5_output(destination):
    if destination is not None and destination.suffix in _HDF5_SUFFIXES:
        message = (
            "Writing the symmetrized structure to an HDF5 file is not implemented."
        )
        raise exception.NotImplemented(message)
