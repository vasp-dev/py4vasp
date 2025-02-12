# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pathlib

import click

import py4vasp
from py4vasp import exception


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
