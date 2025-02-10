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
    check_structure_is_same(structure, STRUCTURE_ZnS, Assert)


def test_read_contcar(tmp_path, Assert):
    filename = tmp_path / "POSCAR"
    with open(filename, "w") as file:
        file.write(str(CONTCAR.from_data(EXAMPLE_CONTCAR)))
    contcar = read.CONTCAR(filename)
    check_contcar_is_same(contcar, EXAMPLE_CONTCAR, Assert)


def check_contcar_is_same(actual, expected, Assert):
    check_structure_is_same(actual.structure, expected.structure, Assert)
    assert actual.system == expected.system
    Assert.allclose(actual.selective_dynamics, expected.selective_dynamics)
    # velocities are written in a lower precision
    Assert.allclose(
        actual.lattice_velocities.astype(np.float32),
        expected.lattice_velocities.astype(np.float32),
    )
    Assert.allclose(
        actual.ion_velocities.astype(np.float32),
        expected.ion_velocities.astype(np.float32),
    )


def check_structure_is_same(actual, expected, Assert):
    check_stoichiometry_is_same(actual.stoichiometry, expected.stoichiometry)
    check_cell_is_same(actual.cell, expected.cell, Assert)
    Assert.allclose(actual.positions, expected.positions)


def check_stoichiometry_is_same(actual, expected):
    assert np.array_equal(actual.number_ion_types, expected.number_ion_types)
    assert np.array_equal(actual.ion_types, expected.ion_types)


def check_cell_is_same(actual, expected, Assert):
    Assert.allclose(actual.lattice_vectors, expected.lattice_vectors)
    Assert.allclose(actual.scale, expected.scale)
