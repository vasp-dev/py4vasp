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
