# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Generate the content of KPOINTS files."""

import numpy as np

from py4vasp import exception
from py4vasp._calculation.symmetry import _SYMPREC
from py4vasp._util import import_

seekpath = import_.optional("seekpath")
spglib = import_.optional("spglib")

# seekpath spells the special points of the high-symmetry path in ASCII, e.g. "GAMMA"
# or "SIGMA_0". VASP reads the label as the remainder of the line, so we can write the
# glyph the label stands for instead.
_GREEK_LETTERS = {"GAMMA": "Γ", "DELTA": "Δ", "LAMBDA": "Λ", "SIGMA": "Σ"}
_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
# the standardized conventional cell is at most as large as the cell it comes from,
# so the determinant of the transformation matrix exceeds 1 only for a supercell
_VOLUME_TOLERANCE = 1e-6


def label_to_unicode(name: str) -> str:
    """Convert the name of a special point to a compact label, e.g. 'SIGMA_0' to 'Σ₀'.

    Names that do not follow seekpath's convention of a point name and an optional
    numeric subscript are passed on unchanged.
    """
    base, separator, subscript = name.partition("_")
    if separator and not subscript.isdigit():
        return name
    return _GREEK_LETTERS.get(base, base) + subscript.translate(_SUBSCRIPTS)


def line_mode(coordinates, labels, number_points, comment: str) -> str:
    """Write the segments of a high-symmetry path as a KPOINTS file in line mode.

    VASP reads the label of a special point from the fourth field of its k-point line,
    so the labels must *not* be hidden behind a comment character.

    Parameters
    ----------
    coordinates : array-like
        Start and end point of every line as fractions of the reciprocal lattice
        vectors, i.e. of shape (number of lines, 2, 3).
    labels : Sequence
        Label of the start and end point of every line.
    number_points : int
        Number of k points VASP generates along every line.
    comment : str
        First line of the file explaining where the path comes from.
    """
    header = f"{comment}\n{number_points}\nline mode\nreciprocal"
    segments = (
        f"{_vector(start)}  {start_label}\n{_vector(end)}  {end_label}"
        for (start, end), (start_label, end_label) in zip(
            coordinates, labels, strict=True
        )
    )
    return "\n".join((header, "\n\n".join(segments)))


def _vector(coordinates) -> str:
    return " ".join(f"{coordinate:12.8f}" for coordinate in coordinates)


def transformation_matrix(cell, symprec: float = _SYMPREC) -> np.ndarray:
    """Determine the matrix transforming *cell* to the standardized conventional cell.

    With spglib's convention (a b c)_standardized = (a b c) P⁻¹ for the basis vectors,
    the fractional coordinates k of a reciprocal-space vector obey k = k_standardized P.
    The matrix is a property of the fractional coordinates alone, so it does not change
    if the cell is rotated.

    Parameters
    ----------
    cell : tuple
        Lattice vectors, direct coordinates, and atomic numbers as spglib expects them.
    symprec : float
        Distance tolerance (in Å) spglib uses to detect the symmetry.

    Returns
    -------
    np.ndarray
        The transformation matrix P.
    """
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    transformation = np.array(dataset.transformation_matrix)
    if np.linalg.det(transformation) > 1 + _VOLUME_TOLERANCE:
        message = (
            "The cell is a supercell of the standardized conventional cell, so the "
            "high-symmetry points and the symmetry-adapted mesh of the crystal do not "
            "apply to it. Please use the primitive cell instead, e.g. by reducing the "
            "structure with `py4vasp symmetrize --primitive`."
        )
        raise exception.IncorrectUsage(message)
    return transformation


def to_input_basis(coordinates, cell, primitive_cell, symprec: float = _SYMPREC):
    """Express k points given in the basis of *primitive_cell* in the basis of *cell*.

    Both cells are mapped onto the standardized conventional cell spglib derives from
    them, so the two conventions cancel. Because that mapping is purely fractional, the
    result is independent of the orientation of either cell.

    Parameters
    ----------
    coordinates : array-like
        k points as fractions of the reciprocal lattice vectors of *primitive_cell*.
    cell : tuple
        The cell in whose basis the k points are expressed, as spglib expects it.
    primitive_cell : tuple
        The cell in whose basis the k points are given, as spglib expects it.
    symprec : float
        Distance tolerance (in Å) spglib uses to detect the symmetry.

    Returns
    -------
    np.ndarray
        The same k points as fractions of the reciprocal lattice vectors of *cell*.
    """
    primitive_transformation = transformation_matrix(primitive_cell, symprec)
    transformation = np.linalg.solve(
        primitive_transformation, transformation_matrix(cell, symprec)
    )
    return np.array(coordinates) @ transformation


def high_symmetry_path(
    cell, number_points: int = 40, time_reversal: bool = True, symprec=_SYMPREC
) -> str:
    """Write the recommended high-symmetry path of a crystal as a KPOINTS file.

    seekpath determines the path following the convention of Hinuma et al. It returns
    the special points in the basis of its own standardized primitive cell, so they are
    mapped back onto the basis of *cell* before they are written to the file.

    Parameters
    ----------
    cell : tuple
        Lattice vectors, direct coordinates, and atomic numbers as spglib expects them.
    number_points : int
        Number of k points VASP generates along every line of the path.
    time_reversal : bool
        Whether the k points are related by time-reversal symmetry.
    symprec : float
        Distance tolerance (in Å) spglib uses to detect the symmetry.

    Returns
    -------
    str
        The content of a KPOINTS file describing the path.
    """
    path = seekpath.get_path(cell, with_time_reversal=time_reversal, symprec=symprec)
    primitive_cell = (
        path["primitive_lattice"],
        path["primitive_positions"],
        path["primitive_types"],
    )
    points = path["point_coords"]
    lines = path["path"]
    coordinates = np.array([[points[first], points[last]] for first, last in lines])
    coordinates = to_input_basis(coordinates, cell, primitive_cell, symprec)
    labels = [
        (label_to_unicode(first), label_to_unicode(last)) for first, last in lines
    ]
    return line_mode(coordinates, labels, number_points, _path_comment(path))


def _path_comment(path) -> str:
    "Document the crystal the path belongs to and whether its band structure folds."
    number_cells = round(path["volume_original_wrt_prim"])
    plural = "" if number_cells == 1 else "s"
    return (
        "k points along high symmetry lines: "
        f"{path['spacegroup_international']} ({path['bravais_lattice_extended']}), "
        f"{number_cells} primitive cell{plural} per unit cell"
    )


def conventional_reciprocal_lattice(lattice_vectors, transformation) -> np.ndarray:
    """Determine the reciprocal lattice vectors of the standardized conventional cell.

    The conventional cell follows from the input one as A_conventional = P⁻ᵀ A, so its
    reciprocal lattice vectors are P inv(A)ᵀ. Deriving the mesh from these vectors
    rather than from the ones of the input cell makes it commensurate with the symmetry
    of the crystal even when the calculation runs in the primitive cell.

    Parameters
    ----------
    lattice_vectors : array-like
        The lattice vectors A of the input cell as rows.
    transformation : array-like
        The transformation matrix P from :func:`transformation_matrix`.

    Returns
    -------
    np.ndarray
        The reciprocal lattice vectors of the conventional cell as rows, without the
        factor 2π.
    """
    return np.array(transformation) @ np.linalg.inv(lattice_vectors).T


def divisions_from_kspacing(reciprocal_lattice, kspacing: float) -> list[int]:
    """Determine the number of k points along every direction for a given spacing.

    Following VASP's KSPACING tag, the number of divisions along direction i is
    2π |b_i| / KSPACING rounded to the nearest integer, with at least one k point along
    every direction. VASP truncates the sum of the ratio and 0.5 to an integer, so a
    ratio of exactly one half rounds up.

    Parameters
    ----------
    reciprocal_lattice : array-like
        The reciprocal lattice vectors as rows, without the factor 2π.
    kspacing : float
        The largest allowed distance between two k points in Å⁻¹.

    Returns
    -------
    list[int]
        The number of k points along the three directions.
    """
    lengths = 2 * np.pi * np.linalg.norm(reciprocal_lattice, axis=1)
    # int(x + 0.5) reproduces the truncation VASP applies instead of numpy's rounding
    # of a half to the nearest even integer
    return [max(1, int(length / kspacing + 0.5)) for length in lengths]
