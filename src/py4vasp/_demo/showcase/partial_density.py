# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Partial charge density of the showcase surface, the states a microscope images."""

import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo.showcase import cell, grid, structure

# Width of the state in Angstrom. It is wider than the charge of a single atom in
# showcase.density, because a state at the Fermi energy reaches into the vacuum, which
# is what lets a tip held above the surface tunnel into it at all.
SURFACE_STATE_WIDTH = 0.55
# How much of the state each of the two carbon sublattices carries. In Bernal-stacked
# graphite the two atoms of a layer are inequivalent: one of them sits directly above an
# atom of the layer below and hybridizes with it, which costs it weight at the Fermi
# energy, while the other faces the centre of a hexagon and keeps it. That is why a
# microscope image of graphite shows a triangular lattice of one maximum per cell rather
# than the honeycomb of the atoms.
ECLIPSED_WEIGHT = 0.25
EXPOSED_WEIGHT = 1.0


def Graphite() -> raw.PartialDensity:
    """Partial charge of the states of the graphite slab near the Fermi energy.

    Summed over bands and k points, which is what VASP writes unless asked to separate
    them, and normalized so that its largest value is one. That matters because py4vasp
    draws its isosurface at an absolute level of 0.2 by default, so the level has to
    mean something without the user picking one.
    """
    lattice_vectors = np.array(cell.Graphite().lattice_vectors)
    counts = grid.grid_for(lattice_vectors)
    positions = np.array(structure.Graphite().positions)
    charge = grid.gaussian_sum(
        positions,
        sublattice_weights(positions),
        np.full(len(positions), SURFACE_STATE_WIDTH),
        lattice_vectors,
        counts,
    )
    charge = charge / np.max(charge)
    return raw.PartialDensity(
        structure=structure.Graphite(),
        # zero stands for "summed" in both cases; separating them is a different kind of
        # calculation and py4vasp refuses to simulate a microscope from one
        bands=_demo.wrap_data([0]),
        kpoints=_demo.wrap_data([0]),
        grid=_demo.wrap_data(counts),
        partial_charge=_demo.wrap_data(_as_vasp_grid(charge)),
    )


def sublattice_weights(positions) -> np.ndarray:
    """The share of the state every atom carries, by the sublattice it sits on.

    An atom is eclipsed when an atom of an adjacent layer sits at the same position in
    the plane, which is what the stacking of graphite does to one atom of every layer.
    """
    in_plane = positions[:, :2]
    heights = positions[:, 2]
    spacing = cell.GRAPHITE_INTERLAYER_DISTANCE / cell.GRAPHITE_HEIGHT
    weights = []
    for position, height in zip(in_plane, heights):
        adjacent = np.isclose(np.abs(heights - height), spacing)
        above_each_other = np.all(np.isclose(in_plane, position), axis=1)
        eclipsed = np.any(adjacent & above_each_other)
        weights.append(ECLIPSED_WEIGHT if eclipsed else EXPOSED_WEIGHT)
    return np.array(weights)


def _as_vasp_grid(field):
    """Order the axes the way VASP writes a partial charge.

    py4vasp transposes the whole array, so the grid axes come last and in the reverse
    order, preceded by the spin component, the band and the k point.
    """
    return field.T[np.newaxis, np.newaxis, np.newaxis]
