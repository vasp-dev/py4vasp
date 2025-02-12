# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import raw
from py4vasp._calculation._CONTCAR import CONTCAR
from py4vasp._calculation.structure import Structure
from py4vasp._raw import read


def test_read_structure(raw_data, tmp_path, Assert):
    raw_structure = raw_data.structure("ZnS")
    filename = tmp_path / "POSCAR"
    with open(filename, "w") as file:
        file.write(str(Structure.from_data(raw_structure)))
    structure = read.structure(filename)
    Assert.same_raw_structure(structure, raw_structure)


def test_read_contcar(raw_data, tmp_path, Assert):
    raw_structure = raw_data.structure("ZnS")
    raw_contcar = raw.CONTCAR(
        structure=raw_structure,
        system="ZnS structure",
        lattice_velocities=raw.VaspData(np.linspace(-1, 1, 9).reshape(3, 3)),
        ion_velocities=raw.VaspData(0.2 - 0.1 * raw_structure.positions),
    )
    filename = tmp_path / "POSCAR"
    with open(filename, "w") as file:
        file.write(str(CONTCAR.from_data(raw_contcar)))
    contcar = read.CONTCAR(filename)
    Assert.same_raw_contcar(contcar, raw_contcar)
