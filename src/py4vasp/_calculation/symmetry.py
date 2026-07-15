# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import exception, raw
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    merge_strings,
    quantity,
)
from py4vasp._util import check, import_

spglib = import_.optional("spglib")

# Tolerance used by spglib when classifying the symmetry operations.
_SYMPREC = 1e-5

# Upper space-group-number boundary of each crystal system.
_CRYSTAL_SYSTEMS = (
    (2, "triclinic"),
    (15, "monoclinic"),
    (74, "orthorhombic"),
    (142, "tetragonal"),
    (167, "trigonal"),
    (194, "hexagonal"),
    (230, "cubic"),
)


def _crystal_system(space_group_number):
    for boundary, name in _CRYSTAL_SYSTEMS:
        if space_group_number <= boundary:
            return name
    message = f"The space group number {space_group_number} is not in the range 1-230."
    raise exception.IncorrectUsage(message)


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

    def has_inversion_symmetry(self) -> bool:
        """Return whether the inversion operation is part of the symmetry group."""
        inversion = -np.eye(3, dtype=int)
        rotations = np.array(self._raw_symmetry.rotations)
        return any(np.array_equal(rotation, inversion) for rotation in rotations)

    def is_symmorphic(self) -> bool:
        """Return whether the space group is symmorphic.

        A space group is symmorphic if an origin exists for which all operations have
        no fractional translation. We use the practical criterion that none of the
        stored operations carries a fractional translation.
        """
        translations = np.array(self._raw_symmetry.translations)
        return bool(np.allclose(translations, 0.0))

    def space_group(self) -> dict:
        """Determine the space group of the crystal from the symmetry operations.

        The space group is deduced with spglib from the symmetry operations VASP
        stored. This requires the optional dependency spglib to be installed.

        Returns
        -------
        dict
            The international space-group number and Hermann-Mauguin symbol, the
            point group, the crystal system, and whether the group is symmorphic.
        """
        space_group_type = self._space_group_type()
        number = space_group_type.number
        return {
            "number": number,
            "international_symbol": space_group_type.international_short,
            "point_group": space_group_type.pointgroup_international,
            "crystal_system": _crystal_system(number),
            "is_symmorphic": self.is_symmorphic(),
        }

    def _space_group_type(self):
        rotations, translations = self._all_operations()
        return spglib.get_spacegroup_type_from_symmetry(
            rotations, translations, lattice=self._lattice(), symprec=_SYMPREC
        )

    def _all_operations(self):
        """Combine the rotations with the pure translations of the primitive cell.

        VASP stores the rotations in the basis of the computational cell. When the
        computational cell contains more than one primitive cell, the pure lattice
        translations relating them are symmetry operations as well and have to be
        added so that spglib classifies the space group correctly.
        """
        rotations = np.array(self._raw_symmetry.rotations)
        translations = np.array(self._raw_symmetry.translations)
        primitive_translations = np.array(self._raw_symmetry.primitive_translations)
        all_rotations = []
        all_translations = []
        for rotation, translation in zip(rotations, translations):
            for primitive_translation in primitive_translations:
                all_rotations.append(rotation)
                all_translations.append((translation + primitive_translation) % 1.0)
        return np.array(all_rotations), np.array(all_translations)

    def _lattice(self):
        cell = self._raw_symmetry.cell
        return self._scale(cell) * np.array(cell.lattice_vectors)

    @staticmethod
    def _scale(cell):
        if check.is_none(cell.scale):
            return 1.0
        return np.array(cell.scale)

    def __str__(self) -> str:
        raw_symmetry = self._raw_symmetry
        lines = [
            f"symmetry group with {int(raw_symmetry.number_of_operations)} operations:"
        ]
        if import_.is_imported(spglib):
            space_group = self.space_group()
            symbol = space_group["international_symbol"]
            lines.append(f"    space group: {symbol} ({space_group['number']})")
            lines.append(f"    crystal system: {space_group['crystal_system']}")
        else:
            lines.append("    space group: not available (requires spglib)")
        inversion = "yes" if self.has_inversion_symmetry() else "no"
        lines.append(f"    inversion symmetry: {inversion}")
        lines.append(
            f"    primitive cells: {int(raw_symmetry.number_of_primitive_cells)}"
        )
        lines.append(f"    ISYM: {int(raw_symmetry.isym)}")
        return "\n".join(lines)


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

    def space_group(self) -> dict:
        """Determine the space group of the crystal from the symmetry operations.

        Check :meth:`SymmetryHandler.space_group` for the description of the returned
        data. Requires the optional dependency spglib.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            SymmetryHandler.space_group,
        )

    def has_inversion_symmetry(self) -> bool:
        """Return whether the inversion operation is part of the symmetry group."""
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            SymmetryHandler.has_inversion_symmetry,
        )

    def __str__(self, selection=None) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            SymmetryHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
