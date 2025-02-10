# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pathlib

import click

import py4vasp


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
    path = path or pathlib.Path.cwd()
    calculation = py4vasp.Calculation.from_path(path)
    print(calculation.structure.to_lammps())
