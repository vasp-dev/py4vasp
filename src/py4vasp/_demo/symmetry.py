# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools

import numpy as np

from py4vasp import _demo, raw

# The demo data below is copied verbatim from real VASP calculations so that the
# derived space groups match crystallographic expectation:
#   CoO (antiferromagnetic, ISPIN=2): F-43m (#216), 24 operations, no inversion
#   AlP (bulk):                       Amm2  (#38),  4 operations, no inversion
# The SrTiO3 symmetry (Pm-3m, #221) is generated from the cubic point group.


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


# Fractional positions of cubic perovskite SrTiO3 in its (primitive = conventional)
# cell: Sr on 1a, Ti on 1b, and three symmetry-equivalent O on 3c.
_SRTIO3_POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0],  # Sr (1a)
        [0.5, 0.5, 0.5],  # Ti (1b)
        [0.0, 0.5, 0.5],  # O  (3c)
        [0.5, 0.0, 0.5],  # O  (3c)
        [0.5, 0.5, 0.0],  # O  (3c)
    ]
)


def SrTiO3():
    """Cubic perovskite SrTiO3 in its conventional cell, space group Pm-3m (#221).

    The full cubic point group (48 signed permutation matrices, all symmorphic) is
    generated and the remaining fields are derived from it, so the data is a
    self-consistent stand-in for the symmetry VASP would report."""
    rotations = _cubic_point_group()
    translations = np.zeros((len(rotations), 3))
    cell = raw.Cell(lattice_vectors=_demo.wrap_data(np.eye(3)), scale=raw.VaspData(4.0))
    return raw.Symmetry(
        cell=cell,
        rotations=_demo.wrap_data(rotations),
        reciprocal_rotations=_demo.wrap_data(_reciprocal_rotations(rotations)),
        translations=_demo.wrap_data(translations),
        inverse_operations=_demo.wrap_data(_inverse_operations(rotations)),
        atom_permutations=_demo.wrap_data(
            [_atom_permutations(rotations, translations, _SRTIO3_POSITIONS)]
        ),
        primitive_lattice_vectors=_demo.wrap_data(4.0 * np.eye(3)),
        primitive_translations=_demo.wrap_data([[0.0, 0.0, 0.0]]),
        number_of_operations=len(rotations),
        number_of_primitive_cells=1,
        isym=2,
        spin_flips=raw.VaspData(None),
    )


def _cubic_point_group():
    """The 48 operations of the cubic point group m-3m as signed permutation matrices."""
    rotations = []
    for permutation in itertools.permutations(range(3)):
        matrix = np.zeros((3, 3), dtype=int)
        for row, column in enumerate(permutation):
            matrix[row, column] = 1
        for signs in itertools.product((1, -1), repeat=3):
            rotations.append(np.diag(signs) @ matrix)
    return np.array(rotations)


def _reciprocal_rotations(rotations):
    """Reciprocal-space rotations, (R^-1)^T for each real-space rotation R."""
    return np.array([np.rint(np.linalg.inv(r).T).astype(int) for r in rotations])


def _inverse_operations(rotations):
    """Fortran 1-based index of the inverse of each (translation-free) operation."""
    inverse = []
    for rotation in rotations:
        target = np.rint(np.linalg.inv(rotation)).astype(int)
        match = next(
            index
            for index, other in enumerate(rotations)
            if np.array_equal(other, target)
        )
        inverse.append(match + 1)
    return inverse


def _atom_permutations(rotations, translations, positions):
    """Fortran 1-based image of each atom under each operation x -> R x + t (mod 1)."""
    permutations = []
    for rotation, translation in zip(rotations, translations):
        images = (positions @ rotation.T + translation) % 1.0
        row = []
        for image in images:
            difference = (positions - image + 0.5) % 1.0 - 0.5
            row.append(int(np.argmin(np.linalg.norm(difference, axis=1))) + 1)
        permutations.append(row)
    return permutations
