# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo.showcase import cell, structure

FORCE_CONSTANT = 6.0  # eV/A^2, a plausible stiffness for an oxide


def cartesian_distortion() -> np.ndarray:
    """Deviation from the ideal positions in Angstrom, shape ``(step, atom, axis)``."""
    lattice_vectors = np.array(cell.Sr2TiO4().lattice_vectors)
    return np.einsum("sai,sij->saj", structure.distortion(), lattice_vectors)


def Sr2TiO4() -> raw.Force:
    """Forces along the showcase relaxation, shrinking as the atoms reach their sites.

    Every force restores the ideal position of its atom, so it points opposite to the
    distortion and vanishes with it. Because the distortion sums to zero over the atoms,
    so do the forces of every step, as Newton's third law requires.
    """
    return raw.Force(
        structure=structure.Sr2TiO4(),
        forces=_demo.wrap_data(-FORCE_CONSTANT * cartesian_distortion()),
    )
