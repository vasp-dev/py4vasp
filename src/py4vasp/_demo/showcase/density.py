# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Charge density of the showcase crystals on their real-space grid."""
import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo.showcase import cell, grid, structure

VALENCE_ELECTRONS = {"C": 4}  # what the PAW potential of VASP treats as valence
CORE_WIDTH = 0.5  # Angstrom, the width of the charge a single atom carries


def Graphite() -> raw.Density:
    """Charge density of the graphite slab.

    The density is a superposition of atom-centred Gaussians carrying the four valence
    electrons of a carbon atom each. It follows the convention VASP writes a density in,
    where the average over the grid is the number of electrons in the cell, so a Bader
    analysis of it reports the electrons of every atom.
    """
    lattice_vectors = np.array(cell.Graphite().lattice_vectors)
    positions = np.array(structure.Graphite().positions)
    electrons = np.full(len(positions), VALENCE_ELECTRONS["C"])
    charge = grid.gaussian_sum(
        positions,
        electrons,
        np.full(len(positions), CORE_WIDTH),
        lattice_vectors,
        grid.grid_for(lattice_vectors),
    )
    volume = np.abs(np.linalg.det(lattice_vectors))
    # VASP multiplies the density by the volume of the cell, so that the mean over the
    # grid is the number of electrons rather than a density
    return raw.Density(
        structure=structure.Graphite(),
        charge=_demo.wrap_data(_as_vasp_grid(charge * volume)),
    )


def _as_vasp_grid(field):
    """Order the axes the way VASP writes a grid and add the spin component.

    A field is evaluated with the first axis running along the first lattice vector,
    while VASP stores the last axis that way, so the array is transposed.
    """
    return field.T[np.newaxis]
