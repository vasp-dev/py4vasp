# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import raw
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    quantity,
)
from py4vasp._util import check


class SymmetryHandler:
    """Handler for the symmetry quantity. Works with exactly one raw.Symmetry object."""

    def __init__(self, raw_symmetry: raw.Symmetry):
        self._raw_symmetry = raw_symmetry

    @classmethod
    def from_data(cls, raw_symmetry: raw.Symmetry) -> "SymmetryHandler":
        return cls(raw_symmetry)

    def read(self) -> dict:
        """Return the symmetry operations and related data as a dictionary.

        The dictionary contains the raw symmetry operations of the crystal in a form
        that is convenient for further processing in Python. The rotation matrices act
        on fractional coordinates of the computational cell; the associated translations
        are given in the same fractional coordinates. Indices (``inverse_operations`` and
        ``atom_permutations``) are converted from the Fortran 1-based convention used in
        the file to 0-based so that they can be used directly to index NumPy arrays.

        Returns
        -------
        dict
            Contains the real- and reciprocal-space rotations, translations, the inverse
            operation of each operation, the atom permutations, the primitive cell
            information, and scalar metadata (number of operations, number of primitive
            cells, ISYM). ``spin_flips`` is ``None`` unless the calculation was
            spin-polarized.
        """
        raw_symmetry = self._raw_symmetry
        spin_flips = raw_symmetry.spin_flips
        return {
            "rotations": np.array(raw_symmetry.rotations),
            "reciprocal_rotations": np.array(raw_symmetry.reciprocal_rotations),
            "translations": np.array(raw_symmetry.translations),
            "inverse_operations": np.array(raw_symmetry.inverse_operations) - 1,
            "atom_permutations": np.array(raw_symmetry.atom_permutations) - 1,
            "spin_flips": None if check.is_none(spin_flips) else np.array(spin_flips),
            "primitive_lattice_vectors": np.array(
                raw_symmetry.primitive_lattice_vectors
            ),
            "primitive_translations": np.array(raw_symmetry.primitive_translations),
            "number_of_operations": int(raw_symmetry.number_of_operations),
            "number_of_primitive_cells": int(raw_symmetry.number_of_primitive_cells),
            "isym": int(raw_symmetry.isym),
        }

    def to_dict(self) -> dict:
        """Public alias for read()."""
        return self.read()


@quantity("symmetry")
class Symmetry:
    """The symmetry operations of the crystal determined by VASP.

    VASP analyzes the crystal structure and stores the set of symmetry operations
    (rotations and translations) it uses to reduce the computational effort. This
    class exposes those operations for further processing and can derive the space
    group of the crystal from them using spglib.
    """

    def __init__(self, source, quantity_name: str = "symmetry"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_symmetry: raw.Symmetry) -> "Symmetry":
        """Create a Symmetry dispatcher from raw data (convenience for testing)."""
        return cls(source=DataSource(raw_symmetry))

    @property
    def path(self):
        """Returns the path from which the output is obtained."""
        return self._path

    def _handler_factory(self, raw_data):
        return SymmetryHandler.from_data(raw_data)

    def read(self) -> dict:
        """Return the symmetry operations and related data as a dictionary.

        Check :meth:`SymmetryHandler.read` for the description of the returned data.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            SymmetryHandler.read,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Public alias for read(). Check that method for the returned data."""
        return self.read()
