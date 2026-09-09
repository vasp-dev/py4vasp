# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Generate the content of KPOINTS files."""

# seekpath spells the special points of the high-symmetry path in ASCII, e.g. "GAMMA"
# or "SIGMA_0". VASP reads the label as the remainder of the line, so we can write the
# glyph the label stands for instead.
_GREEK_LETTERS = {"GAMMA": "Γ", "DELTA": "Δ", "LAMBDA": "Λ", "SIGMA": "Σ"}
_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


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
