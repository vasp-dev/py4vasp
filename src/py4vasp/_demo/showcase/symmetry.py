# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Symmetry operations of the showcase crystals.

The operations are generated from the point group rather than copied from a table, the
way :func:`py4vasp._demo.symmetry.SrTiO3` generates the cubic ones: a transcribed table
of sixteen 3x3 matrices is impossible to review, and generating them makes the claim
that they *are* the point group checkable.
"""
import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo.symmetry import (
    _atom_permutations,
    _inverse_operations,
    _reciprocal_rotations,
)
from py4vasp._demo.showcase import cell, structure

ISYM = 2  # the ISYM setting that makes VASP use symmetry without the charge symmetrizer


def Sr2TiO4() -> raw.Symmetry:
    """Symmetry of Sr2TiO4, space group I4/mmm (139).

    Every atom of the K2NiF4 structure sits on a Wyckoff position that the rotations
    alone leave invariant -- Sr and the apical oxygen on 4e, Ti on 2a, the equatorial
    oxygen on 4c -- so the group is symmorphic and no operation carries a fractional
    translation. The operations belong to the *relaxed* positions, which is why
    :func:`py4vasp._demo.showcase.structure.distortion` decays to exactly zero: a
    trajectory that merely came close would leave the final structure less symmetric
    than the operations claim.
    """
    lattice_vectors = cell.lattice_vectors()
    rotations = _tetragonal_point_group(lattice_vectors)
    translations = np.zeros((len(rotations), 3))
    return raw.Symmetry(
        # the symmetry stores the final cell, at a path of its own, so it is unaffected
        # by the compressed cells of the earlier steps of the trajectory
        cell=raw.Cell(
            lattice_vectors=_demo.wrap_data(lattice_vectors), scale=raw.VaspData(1.0)
        ),
        rotations=_demo.wrap_data(rotations),
        reciprocal_rotations=_demo.wrap_data(_reciprocal_rotations(rotations)),
        translations=_demo.wrap_data(translations),
        inverse_operations=_demo.wrap_data(_inverse_operations(rotations)),
        atom_permutations=_demo.wrap_data(
            [_atom_permutations(rotations, translations, structure.ideal_positions())]
        ),
        primitive_lattice_vectors=_demo.wrap_data(lattice_vectors),
        primitive_translations=_demo.wrap_data([[0.0, 0.0, 0.0]]),
        number_of_operations=len(rotations),
        number_of_primitive_cells=1,
        isym=ISYM,
        spin_flips=raw.VaspData(None),
    )


def _tetragonal_point_group(lattice_vectors) -> np.ndarray:
    """The 16 operations of 4/mmm in the basis of the given lattice vectors.

    VASP reports the rotations in the basis of the computational cell, and the primitive
    cell of a body-centred lattice is not orthogonal -- its first two vectors enclose
    47 degrees for these lattice constants. So the operations are built in Cartesian
    coordinates, where 4/mmm is easy to write down, and then transformed. Every one of
    them comes out integer because the lattice is invariant under the group.
    """
    basis = np.transpose(lattice_vectors)  # columns are the lattice vectors
    rotations = [
        np.linalg.solve(basis, cartesian @ basis)
        for cartesian in _cartesian_operations()
    ]
    return np.rint(rotations).astype(int)


def _cartesian_operations():
    """4/mmm as Cartesian matrices: the four rotations about z, a mirror, an inversion."""
    four_fold = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    mirror_x = np.diag([-1, 1, 1])
    operations = [
        parity * np.linalg.matrix_power(four_fold, power) @ mirror
        for power in range(4)
        for mirror in (np.eye(3, dtype=int), mirror_x)
        for parity in (1, -1)
    ]
    return _unique(operations)


def _unique(operations):
    unique = []
    for operation in operations:
        if not any(np.array_equal(operation, other) for other in unique):
            unique.append(operation)
    return unique
