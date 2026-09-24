# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""The mass of every element, used to undo the mass weighting VASP applies."""

import numpy as np

from py4vasp import exception

#: Standard atomic weight of every element in atomic mass units, in the order of the
#: atomic number. These are the conventional values IUPAC reports for the isotope
#: mixture found on earth; for the elements without a stable isotope it is the mass
#: number of the longest-lived one. VASP uses the POMASS of the POTCAR instead, which
#: agrees with these values to a few parts in ten thousand unless you overwrite it, for
#: example to study an isotope.
# fmt: off
TABLE = {
    "H": 1.008, "He": 4.002602, "Li": 6.94, "Be": 9.012183,
    "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.9984, "Ne": 20.1797, "Na": 22.98977, "Mg": 24.305,
    "Al": 26.98154, "Si": 28.085, "P": 30.97376, "S": 32.06,
    "Cl": 35.45, "Ar": 39.948, "K": 39.0983, "Ca": 40.078,
    "Sc": 44.95591, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961,
    "Mn": 54.93804, "Fe": 55.845, "Co": 58.93319, "Ni": 58.6934,
    "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.63,
    "As": 74.92159, "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.4678, "Sr": 87.62, "Y": 88.90584, "Zr": 91.224,
    "Nb": 92.90637, "Mo": 95.95, "Tc": 97.90721, "Ru": 101.07,
    "Rh": 102.9055, "Pd": 106.42, "Ag": 107.8682, "Cd": 112.414,
    "In": 114.818, "Sn": 118.71, "Sb": 121.76, "Te": 127.6,
    "I": 126.9045, "Xe": 131.293, "Cs": 132.9055, "Ba": 137.327,
    "La": 138.9055, "Ce": 140.116, "Pr": 140.9077, "Nd": 144.242,
    "Pm": 144.9128, "Sm": 150.36, "Eu": 151.964, "Gd": 157.25,
    "Tb": 158.9254, "Dy": 162.5, "Ho": 164.9303, "Er": 167.259,
    "Tm": 168.9342, "Yb": 173.054, "Lu": 174.9668, "Hf": 178.49,
    "Ta": 180.9479, "W": 183.84, "Re": 186.207, "Os": 190.23,
    "Ir": 192.217, "Pt": 195.084, "Au": 196.9666, "Hg": 200.592,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.9804, "Po": 208.9824,
    "At": 209.9872, "Rn": 222.0176, "Fr": 223.0197, "Ra": 226.0254,
    "Ac": 227.0277, "Th": 232.0377, "Pa": 231.0359, "U": 238.0289,
    "Np": 237.0482, "Pu": 244.0642, "Am": 243.0614, "Cm": 247.0703,
    "Bk": 247.0703, "Cf": 251.0796, "Es": 252.083, "Fm": 257.0951,
    "Md": 258.0984, "No": 259.101, "Lr": 262.11, "Rf": 267.122,
    "Db": 268.126, "Sg": 271.134, "Bh": 270.133, "Hs": 269.1338,
    "Mt": 278.156, "Ds": 281.165, "Rg": 281.166, "Cn": 285.177,
    "Nh": 286.182, "Fl": 289.19, "Mc": 289.194, "Lv": 293.204,
    "Ts": 293.208, "Og": 294.214,
}
# fmt: on


def of(elements) -> np.ndarray:
    """Look up the mass of every given element.

    Parameters
    ----------
    elements : Sequence[str]
        The chemical symbol of every atom, e.g. the "elements" that
        :py:meth:`py4vasp._calculation.structure.Structure.read` reports.

    Returns
    -------
    np.ndarray
        The mass of every element in atomic mass units in the given order.
    """
    return np.array([_single_element(element) for element in elements])


def _single_element(element):
    try:
        return TABLE[element]
    except KeyError as error:
        message = (
            f"py4vasp does not know the mass of the element {element!r}. Please check "
            "that the elements of the structure are set correctly; VASP does not write "
            "them to the HDF5 file if the POTCAR is missing."
        )
        raise exception.IncorrectUsage(message) from error
