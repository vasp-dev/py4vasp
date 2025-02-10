# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import click


@click.group()
def cli():
    pass


@cli.command()
@click.argument("source", type=click.Choice(("structure",)), metavar="SOURCE")
@click.argument("destination", type=click.STRING)
def convert(source):
    """Convert a quantity to a different format.

    SOURCE is the name of the quantity that you want to convert.
    DESTINATION is the target format to which you want to convert.
    """
    pass
