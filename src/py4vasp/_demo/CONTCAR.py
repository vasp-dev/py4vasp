# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def Sr2TiO4():
    structure = _demo.structure.Sr2TiO4()
    structure.cell.lattice_vectors = structure.cell.lattice_vectors[-1]
    structure.positions = structure.positions[-1]
    return raw.CONTCAR(structure=structure, system=b"Sr2TiO4")


def Fe3O4():
    structure = _demo.structure.Fe3O4()
    structure.cell.lattice_vectors = structure.cell.lattice_vectors[-1]
    structure.positions = structure.positions[-1]
    even_numbers = np.arange(structure.positions.size) % 2 == 0
    selective_dynamics = even_numbers.reshape(structure.positions.shape)
    lattice_velocities = 0.1 * structure.cell.lattice_vectors**2 - 0.3
    shape = structure.positions.shape
    ion_velocities = np.sqrt(np.arange(np.prod(shape)).reshape(shape))
    return raw.CONTCAR(
        structure=structure,
        system="Fe3O4",
        selective_dynamics=raw.VaspData(selective_dynamics),
        lattice_velocities=raw.VaspData(lattice_velocities),
        ion_velocities=raw.VaspData(ion_velocities),
    )
