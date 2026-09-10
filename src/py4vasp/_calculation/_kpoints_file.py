# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Generate the content of KPOINTS files."""

import warnings

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
    if number_points < 1:
        message = (
            "VASP samples at least one k point along every line of the path, but "
            f"number_points is {number_points}."
        )
        raise exception.IncorrectUsage(message)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = seekpath.get_path(
            cell, with_time_reversal=time_reversal, symprec=symprec
        )
    _report_warnings_of_seekpath(caught)
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
    return line_mode(coordinates, labels, number_points, path_comment(path))


def _report_warnings_of_seekpath(caught):
    """Give the ambiguities seekpath reports the context a user needs.

    seekpath warns when a lattice sits close to a more symmetric one, which decides
    which path it picks. Its bare warning names the coincidence but not what to do
    about it. Everything else -- spglib emits a DeprecationWarning on every call -- is
    passed on unchanged, so that we do not cry wolf.
    """
    for warning in caught:
        if issubclass(warning.category, seekpath.hpkot.EdgeCaseWarning):
            message = (
                f"seekpath reports that the lattice is a borderline case: "
                f"{warning.message} It resolves the ambiguity one way, so the path may "
                "not be the one you expect. Check the space group in the first line of "
                "the file, and pass a different symprec if your cell is only "
                "approximately symmetric."
            )
            warnings.warn(message, UserWarning)
        else:
            warnings.warn(warning.message, warning.category)


def path_comment(path) -> str:
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


def generating_lattice(transformation, divisions) -> np.ndarray:
    """Determine the basis vectors of the k-point mesh in the basis of the input cell.

    The mesh subdivides the reciprocal lattice vectors of the conventional cell, i.e.
    its basis vectors are b_i(conventional) / n_i. Expressing them in the reciprocal
    basis of the input cell cancels the lattice vectors, so all that remains is dividing
    the rows of the transformation matrix by the number of divisions.

    Parameters
    ----------
    transformation : array-like
        The transformation matrix P from :func:`transformation_matrix`.
    divisions : Sequence[int]
        The number of k points along the three directions of the conventional cell.

    Returns
    -------
    np.ndarray
        The three basis vectors of the mesh as fractions of the reciprocal lattice
        vectors of the input cell.
    """
    return np.array(transformation) / np.reshape(divisions, (3, 1))


def generating_lattice_mode(vectors, shift, comment: str) -> str:
    """Write a k-point mesh as a KPOINTS file defined by three basis vectors.

    Setting the number of k points to 0 makes VASP generate the mesh automatically from
    the three basis vectors that follow. "Reduced" declares that they are given as
    fractions of the reciprocal lattice vectors.

    Parameters
    ----------
    vectors : array-like
        The three basis vectors of the mesh as rows.
    shift : array-like
        Shift of the mesh in the same basis as the vectors.
    comment : str
        First line of the file explaining how the mesh was generated.

    Returns
    -------
    str
        The content of a KPOINTS file describing the mesh.
    """
    rows = [_vector(vector) for vector in (*vectors, shift)]
    return "\n".join((comment, "0", "Reduced", *rows))


def mesh_comment(divisions, kspacing: float | None = None) -> str:
    "Document the mesh of the conventional cell the k points are generated from."
    comment = "k mesh of the conventional cell: divisions " + " ".join(
        str(division) for division in divisions
    )
    if kspacing is None:
        return comment
    return f"{comment}, kspacing {kspacing}"


def regular_mesh(
    cell, kspacing=None, divisions=None, shift=None, symprec=_SYMPREC
) -> str:
    """Write a mesh commensurate with the symmetry of the crystal as a KPOINTS file.

    The mesh subdivides the conventional cell of the crystal, so it retains the full
    symmetry of the lattice even when the calculation runs in the primitive cell. Give
    either the *kspacing* determining the density of the mesh or the *divisions* of the
    conventional cell explicitly.

    Parameters
    ----------
    cell : tuple
        Lattice vectors, direct coordinates, and atomic numbers as spglib expects them.
    kspacing : float
        The largest allowed distance between two k points in Å⁻¹.
    divisions : Sequence[int]
        The number of k points along the three directions of the conventional cell.
    shift : array-like
        Shift of the mesh as fractions of its basis vectors, defaults to no shift.
    symprec : float
        Distance tolerance (in Å) spglib uses to detect the symmetry.

    Returns
    -------
    str
        The content of a KPOINTS file describing the mesh.
    """
    _raise_if_mesh_not_valid(kspacing, divisions, shift)
    transformation = transformation_matrix(cell, symprec)
    if divisions is None:
        reciprocal_lattice = conventional_reciprocal_lattice(cell[0], transformation)
        divisions = divisions_from_kspacing(reciprocal_lattice, kspacing)
    vectors = generating_lattice(transformation, divisions)
    shift = (0, 0, 0) if shift is None else shift
    return generating_lattice_mode(vectors, shift, mesh_comment(divisions, kspacing))


def _raise_if_mesh_not_valid(kspacing, divisions, shift):
    """Reject arguments that would end up as nan, inf, or a malformed line in the file.

    Without these checks a spacing of zero raises an OverflowError, a division of zero
    divides by zero, and a shift with the wrong number of elements silently produces a
    line VASP cannot read.
    """
    if kspacing is None and divisions is None:
        message = (
            "Please specify how dense the mesh should be, either with the kspacing "
            "argument or with the divisions of the conventional cell."
        )
        raise exception.IncorrectUsage(message)
    if kspacing is not None and divisions is not None:
        message = (
            "The kspacing and the divisions both set the density of the mesh, so "
            "please specify only one of them."
        )
        raise exception.IncorrectUsage(message)
    if divisions is None:
        if kspacing <= 0:
            message = (
                "The spacing of the k points is a distance in the reciprocal space, so "
                f"it must be positive, but it is {kspacing}."
            )
            raise exception.IncorrectUsage(message)
    else:
        _raise_if_not_three_elements(divisions, "divisions of the mesh")
        if any(division < 1 for division in divisions):
            message = (
                "Every direction is sampled by at least one k point, but the divisions "
                f"of the mesh are {list(divisions)}."
            )
            raise exception.IncorrectUsage(message)
    if shift is not None:
        _raise_if_not_three_elements(shift, "shift of the mesh")


def _raise_if_not_three_elements(values, description):
    if np.size(values) != 3:
        message = (
            f"The {description} must have three elements, one for every direction, but "
            f"{np.size(values)} were given."
        )
        raise exception.IncorrectUsage(message)
