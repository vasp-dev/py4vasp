# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._util import parse


def structure(filename, ion_types=None):
    contcar = CONTCAR(filename, ion_types)
    return contcar.structure


def CONTCAR(filename, ion_types=None):
    with open(filename, "r") as file:
        return parse.POSCAR(file.read(), ion_types)
