# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw

# The demo data below is copied verbatim from real VASP calculations so that the
# derived space groups match crystallographic expectation:
#   CoO (antiferromagnetic, ISPIN=2): F-43m (#216), 24 operations, no inversion
#   AlP (bulk):                       Amm2  (#38),  4 operations, no inversion


def CoO():
    """Antiferromagnetic CoO in a single primitive (rhombohedral FCC) cell."""
    rotations = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        [[-1, -1, -1], [0, 0, 1], [1, 0, 0]],
        [[-1, -1, -1], [0, 1, 0], [0, 0, 1]],
        [[-1, -1, -1], [1, 0, 0], [0, 1, 0]],
        [[0, 1, 0], [1, 0, 0], [-1, -1, -1]],
        [[1, 0, 0], [0, 0, 1], [-1, -1, -1]],
        [[0, 0, 1], [0, 1, 0], [-1, -1, -1]],
        [[0, 0, 1], [-1, -1, -1], [0, 1, 0]],
        [[0, 1, 0], [-1, -1, -1], [1, 0, 0]],
        [[1, 0, 0], [-1, -1, -1], [0, 0, 1]],
        [[-1, -1, -1], [0, 1, 0], [1, 0, 0]],
        [[-1, -1, -1], [1, 0, 0], [0, 0, 1]],
        [[-1, -1, -1], [0, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [0, 0, 1], [-1, -1, -1]],
        [[1, 0, 0], [0, 1, 0], [-1, -1, -1]],
        [[0, 0, 1], [1, 0, 0], [-1, -1, -1]],
        [[1, 0, 0], [-1, -1, -1], [0, 1, 0]],
        [[0, 0, 1], [-1, -1, -1], [1, 0, 0]],
        [[0, 1, 0], [-1, -1, -1], [0, 0, 1]],
        [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
    ]
    reciprocal_rotations = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        [[0, -1, 0], [0, -1, 1], [1, -1, 0]],
        [[-1, 0, 0], [-1, 1, 0], [-1, 0, 1]],
        [[0, 0, -1], [1, 0, -1], [0, 1, -1]],
        [[0, 1, -1], [1, 0, -1], [0, 0, -1]],
        [[1, -1, 0], [0, -1, 1], [0, -1, 0]],
        [[-1, 0, 1], [-1, 1, 0], [-1, 0, 0]],
        [[-1, 0, 1], [-1, 0, 0], [-1, 1, 0]],
        [[0, 1, -1], [0, 0, -1], [1, 0, -1]],
        [[1, -1, 0], [0, -1, 0], [0, -1, 1]],
        [[0, 0, -1], [0, 1, -1], [1, 0, -1]],
        [[0, -1, 0], [1, -1, 0], [0, -1, 1]],
        [[-1, 0, 0], [-1, 0, 1], [-1, 1, 0]],
        [[-1, 1, 0], [-1, 0, 1], [-1, 0, 0]],
        [[1, 0, -1], [0, 1, -1], [0, 0, -1]],
        [[0, -1, 1], [1, -1, 0], [0, -1, 0]],
        [[1, 0, -1], [0, 0, -1], [0, 1, -1]],
        [[0, -1, 1], [0, -1, 0], [1, -1, 0]],
        [[-1, 1, 0], [-1, 0, 0], [-1, 0, 1]],
        [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
    ]
    number_of_operations = 24
    inverse_operations = [
        1,
        3,
        2,
        10,
        5,
        16,
        7,
        19,
        13,
        4,
        18,
        12,
        9,
        21,
        15,
        6,
        17,
        11,
        8,
        20,
        14,
        22,
        23,
        24,
    ]
    atom_permutations = [number_of_operations * [[1, 2]]]
    cell = raw.Cell(
        lattice_vectors=_demo.wrap_data(
            [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
        ),
        scale=raw.VaspData(4.4479601811386305),
    )
    return raw.Symmetry(
        cell=cell,
        rotations=_demo.wrap_data(rotations),
        reciprocal_rotations=_demo.wrap_data(reciprocal_rotations),
        translations=_demo.wrap_data(np.zeros((number_of_operations, 3))),
        inverse_operations=_demo.wrap_data(inverse_operations),
        atom_permutations=_demo.wrap_data(atom_permutations),
        primitive_lattice_vectors=_demo.wrap_data(
            2.2239800905693152
            * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        ),
        primitive_translations=_demo.wrap_data([[0.0, 0.0, 0.0]]),
        number_of_operations=number_of_operations,
        number_of_primitive_cells=1,
        isym=2,
        spin_flips=_demo.wrap_data(np.ones((1, number_of_operations))),
    )


def AlP():
    """Bulk AlP in a computational cell containing two primitive cells."""
    rotations = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
    ]
    number_of_operations = 4
    atom_permutations = [
        number_of_operations * [[1, 2, 3, 4]],
        number_of_operations * [[2, 1, 4, 3]],
    ]
    cell = raw.Cell(
        lattice_vectors=_demo.wrap_data(np.eye(3)),
        scale=raw.VaspData(4.048183),
    )
    return raw.Symmetry(
        cell=cell,
        rotations=_demo.wrap_data(rotations),
        reciprocal_rotations=_demo.wrap_data(rotations),
        translations=_demo.wrap_data(np.zeros((number_of_operations, 3))),
        inverse_operations=_demo.wrap_data([1, 2, 3, 4]),
        atom_permutations=_demo.wrap_data(atom_permutations),
        primitive_lattice_vectors=_demo.wrap_data(
            [
                [2.0240915, -2.0240915, 0.0],
                [2.0240915, 2.0240915, 0.0],
                [0.0, 0.0, 4.048183],
            ]
        ),
        primitive_translations=_demo.wrap_data([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]),
        number_of_operations=number_of_operations,
        number_of_primitive_cells=2,
        isym=2,
        spin_flips=raw.VaspData(None),
    )
