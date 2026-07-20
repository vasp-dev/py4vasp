# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses

import numpy as np

from py4vasp import exception, raw
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    merge_strings,
    merge_to_database,
    quantity,
)
from py4vasp._raw.models import SymmetryModel
from py4vasp._util import check, import_

spglib = import_.optional("spglib")


@dataclasses.dataclass
class SpaceGroup:
    """The space group of a crystal and related classification."""

    number: int
    "The international space-group number (1-230)."
    international_symbol: str
    "The Hermann-Mauguin (international short) symbol, e.g. F-43m."
    point_group: str
    "The point group in international notation, e.g. -43m."
    crystal_system: str
    "The crystal system, e.g. cubic or orthorhombic."
    is_symmorphic: bool
    "Whether the space group is symmorphic."


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

# Crystal-family letter of the Pearson symbol for each crystal system.
_FAMILY_LETTER = {
    "triclinic": "a",
    "monoclinic": "m",
    "orthorhombic": "o",
    "tetragonal": "t",
    "trigonal": "h",
    "hexagonal": "h",
    "cubic": "c",
}

# Number of lattice points in the conventional cell for each centering.
_CENTERING_MULTIPLICITY = {"P": 1, "S": 2, "I": 2, "F": 4, "R": 3}


def _crystal_system(space_group_number):
    for boundary, name in _CRYSTAL_SYSTEMS:
        if space_group_number <= boundary:
            return name
    message = f"The space group number {space_group_number} is not in the range 1-230."
    raise exception.IncorrectUsage(message)


def _centering(space_group_type):
    """Return the centering letter (P, S, I, F, R) of the conventional cell.

    The base-centered lattices (A, B, C) are unified into the single symbol S.
    """
    letter = space_group_type.international_short[0]
    return "S" if letter in "ABC" else letter


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
            cells, ISYM). ``spin_flips`` is only present for spin-polarized calculations.
        """
        raw_symmetry = self._raw_symmetry
        result = {
            "rotations": np.array(raw_symmetry.rotations),
            "reciprocal_rotations": np.array(raw_symmetry.reciprocal_rotations),
            "translations": np.array(raw_symmetry.translations),
            "inverse_operations": np.array(raw_symmetry.inverse_operations) - 1,
            "atom_permutations": np.array(raw_symmetry.atom_permutations) - 1,
            "primitive_lattice_vectors": np.array(
                raw_symmetry.primitive_lattice_vectors
            ),
            "primitive_translations": np.array(raw_symmetry.primitive_translations),
            "number_of_operations": int(raw_symmetry.number_of_operations[()]),
            "number_of_primitive_cells": int(raw_symmetry.number_of_primitive_cells[()]),
            "isym": int(raw_symmetry.isym),
        }
        if not check.is_none(raw_symmetry.spin_flips):
            result["spin_flips"] = np.array(raw_symmetry.spin_flips)
        return result

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

    def space_group(self) -> SpaceGroup:
        """Determine the space group of the crystal from the symmetry operations.

        The space group is deduced with spglib from the symmetry operations VASP
        stored.

        Returns
        -------
        SpaceGroup
            The international space-group number and Hermann-Mauguin symbol, the
            point group, the crystal system, and whether the group is symmorphic.
        """
        space_group_type = self._space_group_type()
        number = space_group_type.number
        return SpaceGroup(
            number=number,
            international_symbol=space_group_type.international_short,
            point_group=space_group_type.pointgroup_international,
            crystal_system=_crystal_system(number),
            is_symmorphic=self.is_symmorphic(),
        )

    def to_database(self) -> SymmetryModel:
        """Serialize the symmetry data for database storage.

        The stored quantities deliberately reduce the symmetry operations to a few
        scalars that make it easy to search for a calculation (space group, crystal
        system, presence of inversion symmetry, ...). Space-group information requires
        spglib; if it is not installed those fields are left empty.
        """
        if import_.is_imported(spglib):
            space_group_type = self._space_group_type()
            number = space_group_type.number
            space_group = number
            space_group_symbol = space_group_type.international_short
            crystal_system = _crystal_system(number)
            point_group_schoenflies = space_group_type.pointgroup_schoenflies
            bravais_lattice = self._bravais_lattice(space_group_type)
            pearson_symbol = f"{bravais_lattice}{self._number_of_conventional_atoms(bravais_lattice)}"
        else:
            space_group = space_group_symbol = crystal_system = None
            point_group_schoenflies = bravais_lattice = pearson_symbol = None
        return SymmetryModel(
            space_group=space_group,
            space_group_symbol=space_group_symbol,
            crystal_system=crystal_system,
            point_group_schoenflies=point_group_schoenflies,
            bravais_lattice=bravais_lattice,
            pearson_symbol=pearson_symbol,
            has_inversion_symmetry=self.has_inversion_symmetry(),
            number_of_operations=int(self._raw_symmetry.number_of_operations),
            number_of_primitive_cells=int(self._raw_symmetry.number_of_primitive_cells),
            is_symmorphic=self.is_symmorphic(),
        )

    def point_group_schoenflies(self) -> str:
        """Return the point group of the crystal in Schoenflies notation, e.g. Td."""
        return self._space_group_type().pointgroup_schoenflies

    def bravais_lattice(self) -> str:
        """Return the two-letter Bravais-lattice symbol, e.g. cF, oS, or hP.

        The first letter denotes the crystal family (a, m, o, t, h, c) and the second
        the centering (P, S, I, F, R). There are 14 possible combinations.
        """
        return self._bravais_lattice(self._space_group_type())

    def pearson_symbol(self) -> str:
        """Return the Pearson symbol, e.g. cF8.

        The Pearson symbol combines the Bravais-lattice symbol with the number of atoms
        in the conventional cell.
        """
        bravais_lattice = self._bravais_lattice(self._space_group_type())
        number_atoms = self._number_of_conventional_atoms(bravais_lattice)
        return f"{bravais_lattice}{number_atoms}"

    def _bravais_lattice(self, space_group_type):
        family = _FAMILY_LETTER[_crystal_system(space_group_type.number)]
        return family + _centering(space_group_type)

    def _number_of_conventional_atoms(self, bravais_lattice):
        multiplicity = _CENTERING_MULTIPLICITY[bravais_lattice[1]]
        number_atoms = np.array(self._raw_symmetry.atom_permutations).shape[-1]
        number_primitive_atoms = number_atoms // int(
            self._raw_symmetry.number_of_primitive_cells
        )
        return number_primitive_atoms * multiplicity

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
            symbol = space_group.international_symbol
            lines.append(f"    space group: {symbol} ({space_group.number})")
            lines.append(f"    crystal system: {space_group.crystal_system}")
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

    VASP analyzes the crystal structure and determines the symmetry operations
    (rotations and translations) that leave it invariant. This class exposes those
    operations for further processing and derives common crystallographic descriptors
    such as the space group, the Bravais lattice, and the Pearson symbol from them
    using spglib.

    Examples
    --------
    First, we create some example data so that you can follow along. Please define a
    variable `path` with the path to a directory that does not contain any VASP
    calculation data. Alternatively, use your own data if you have run VASP.

    >>> from py4vasp import demo
    >>> calculation = demo.calculation(path)

    Read the symmetry operations into a Python dictionary for further processing

    >>> calculation.symmetry.read()
    {'rotations': array(...), ..., 'isym': 2, 'spin_flips': array(...)}

    Check whether the crystal is centrosymmetric

    >>> calculation.symmetry.has_inversion_symmetry()
    False
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
        """Read the symmetry operations into a dictionary.

        The dictionary provides the complete set of symmetry operations as NumPy
        arrays for convenient postprocessing. The rotation matrices act on fractional
        coordinates of the computational cell and the translations are given in the
        same basis. The index arrays ``inverse_operations`` and ``atom_permutations``
        are converted from the Fortran 1-based convention used in the file to 0-based,
        so that they index NumPy arrays directly. ``spin_flips`` is only present for
        spin-polarized calculations.

        Returns
        -------
        dict
            The real- and reciprocal-space rotations, the translations, the inverse of
            each operation, the atom permutations, the primitive-cell lattice vectors
            and translations, and scalar metadata (number of operations, number of
            primitive cells, and the ISYM setting).

        Examples
        --------
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        Read the symmetry operations into a Python dictionary

        >>> calculation.symmetry.read()
        {'rotations': array(...), ..., 'isym': 2, 'spin_flips': array(...)}
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            SymmetryHandler.read,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Convenient alias for :py:meth:`read`. Please read the documentation there."""
        return self.read()

    def space_group(self) -> SpaceGroup:
        """Determine the space group of the crystal from its symmetry operations.

        The symmetry operations are classified with spglib to identify the space group
        the crystal belongs to.

        Returns
        -------
        SpaceGroup
            The international space-group number and Hermann-Mauguin symbol, the point
            group, the crystal system, and whether the space group is symmorphic.

        Examples
        --------
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        >>> calculation.symmetry.space_group()
        SpaceGroup(number=216, international_symbol='F-43m', point_group='-43m', crystal_system='cubic', is_symmorphic=True)
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            SymmetryHandler.space_group,
        )

    def has_inversion_symmetry(self) -> bool:
        """Check whether the inversion operation is part of the symmetry group.

        Returns
        -------
        bool
            True if the crystal is centrosymmetric, i.e. the inversion maps the crystal
            onto itself.

        Examples
        --------
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        >>> calculation.symmetry.has_inversion_symmetry()
        False
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            SymmetryHandler.has_inversion_symmetry,
        )

    def point_group_schoenflies(self) -> str:
        """Determine the point group of the crystal in Schoenflies notation.

        Returns
        -------
        str
            The point group in Schoenflies notation, e.g. Td.

        Examples
        --------
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        >>> calculation.symmetry.point_group_schoenflies()
        'Td'
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            SymmetryHandler.point_group_schoenflies,
        )

    def bravais_lattice(self) -> str:
        """Determine the Bravais lattice of the crystal.

        Returns
        -------
        str
            The two-letter Bravais-lattice symbol, one of the 14 possible combinations
            of crystal family (a, m, o, t, h, c) and centering (P, S, I, F, R), e.g. cF.

        Examples
        --------
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        >>> calculation.symmetry.bravais_lattice()
        'cF'
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            SymmetryHandler.bravais_lattice,
        )

    def pearson_symbol(self) -> str:
        """Determine the Pearson symbol of the crystal.

        Returns
        -------
        str
            The Pearson symbol, combining the Bravais-lattice symbol with the number of
            atoms in the conventional cell, e.g. cF8.

        Examples
        --------
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        >>> calculation.symmetry.pearson_symbol()
        'cF8'
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            SymmetryHandler.pearson_symbol,
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

    def _to_database(self) -> dict:
        """Return {quantity[_selection]: handler_result} for database storage."""
        return merge_to_database(
            self._source,
            self._quantity_name,
            SymmetryHandler.from_data,
            SymmetryHandler.to_database,
        )
