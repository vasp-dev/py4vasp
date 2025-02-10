# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import raw
from py4vasp._calculation._CONTCAR import CONTCAR
from py4vasp._calculation.structure import Structure
from py4vasp._raw import read

STRUCTURE_ZnS = raw.Structure(
    raw.Stoichiometry(number_ion_types=[2, 2], ion_types=["Zn", "S"]),
    raw.Cell(
        lattice_vectors=np.array([[1.9, -3.3, 0.0], [1.9, 3.3, 0.0], [0, 0, 6.2]]),
        scale=raw.VaspData(1.0),
    ),
    positions=np.array(
        [
            [1 / 3, 2 / 3, 0.0],
            [2 / 3, 1 / 3, 0.5],
            [1 / 3, 2 / 3, 0.375],
            [2 / 3, 1 / 3, 0.875],
        ]
    ),
)
EXAMPLE_CONTCAR = raw.CONTCAR(
    structure=STRUCTURE_ZnS,
    system="ZnS structure",
    lattice_velocities=raw.VaspData(np.linspace(-1, 1, 9).reshape(3, 3)),
    ion_velocities=raw.VaspData(0.2 - 0.1 * STRUCTURE_ZnS.positions),
)


def test_read_structure(tmp_path, Assert):
    filename = tmp_path / "POSCAR"
    with open(filename, "w") as file:
        file.write(str(Structure.from_data(STRUCTURE_ZnS)))
    structure = read.structure(filename)
    Assert.same_raw_structure(structure, STRUCTURE_ZnS)


def test_read_contcar(tmp_path, Assert):
    filename = tmp_path / "POSCAR"
    with open(filename, "w") as file:
        file.write(str(CONTCAR.from_data(EXAMPLE_CONTCAR)))
    contcar = read.CONTCAR(filename)
    Assert.same_raw_contcar(contcar, EXAMPLE_CONTCAR)
