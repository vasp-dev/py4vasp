# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Work function of the showcase surface: the potential across the slab."""
import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo.showcase import cell, grid, structure

Z_DIRECTION = 3  # the IDIPOL setting that averages over the planes normal to z
# The plane-averaged potential is not a local function of the density: it is the density
# convolved with the Coulomb interaction, which reaches much further than an atom. So it
# is modelled with a width several times the one showcase.density uses, which is what
# keeps the interior of the slab at one depth instead of dipping once per layer.
POTENTIAL_WIDTH = 1.2  # Angstrom
POTENTIAL_DEPTH = 18.0  # eV below the vacuum level in the middle of the slab
# The work function of highly oriented pyrolytic graphite as the literature reports it.
# The vacuum level is the zero of the energy axis, which is the convention a work
# function is quoted in, so the Fermi energy lies this far below zero.
WORK_FUNCTION = 4.6  # eV
FERMI_ENERGY = -WORK_FUNCTION


def Graphite() -> raw.Workfunction:
    """Plane-averaged potential across the graphite slab.

    The potential is flat in the vacuum, drops at each surface and stays at one depth
    through the slab, which is the shape a work function is read off. The vacuum level
    is the zero of the energy axis, so the work function is the Fermi energy with the
    sign reversed.
    """
    distance, potential = _profile()
    vacuum_level = np.max(potential)
    return raw.Workfunction(
        idipol=Z_DIRECTION,
        distance=_demo.wrap_data(distance),
        average_potential=_demo.wrap_data(potential),
        # a slab centred in its cell exposes the same surface on both sides, so the
        # vacuum it faces is at the same level in both directions
        vacuum_potential=_demo.wrap_data(np.full(2, vacuum_level)),
        reference_potential=_reference_potential(),
        fermi_energy=FERMI_ENERGY,
    )


def _reference_potential():
    # imported here because the band gap of the slab is quoted relative to the Fermi
    # energy defined above, so the two modules would otherwise import each other
    from py4vasp._demo.showcase import bandgap

    return bandgap.Graphite()


def _profile():
    """Distance along the third lattice vector and the potential averaged over it.

    The plane average of a sum of Gaussians centred on the atoms is a sum of Gaussians
    in the remaining direction, because integrating one over its plane gives one. So
    this needs no three-dimensional grid, only the heights of the atoms.
    """
    height = cell.GRAPHITE_HEIGHT
    number_points = grid.grid_for(np.array(cell.Graphite().lattice_vectors))[2]
    # the same points the grid quantities are sampled on, so the profiles line up
    distance = np.arange(number_points) / number_points * height
    heights = np.array(structure.Graphite().positions)[:, 2] * height
    smeared = _smear(distance, heights, height)
    # the vacuum level is the zero of the energy axis and the potential is deepest
    # where the atoms are, so the shape of the density is inverted and shifted
    potential = -POTENTIAL_DEPTH * smeared / np.max(smeared)
    return distance, potential - np.max(potential)


def _smear(distance, heights, period):
    """Sum of periodic Gaussians in one dimension, one per atom."""
    # three periods cover every image that a Gaussian of this width still reaches
    images = np.arange(-1, 2) * period
    offset = distance[:, np.newaxis, np.newaxis] - heights[:, np.newaxis] + images
    return np.sum(np.exp(-0.5 * (offset / POTENTIAL_WIDTH) ** 2), axis=(1, 2))
