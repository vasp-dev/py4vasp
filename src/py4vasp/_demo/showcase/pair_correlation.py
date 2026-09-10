# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Pair-correlation function of the showcase crystals, from their neighbour shells."""
import functools
import itertools

import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo import showcase
from py4vasp._demo.showcase import cell, structure
from py4vasp._util import convert

MAX_DISTANCE = 8.0  # Angstrom, far enough to show the crystal settling around one
WIDTH = 0.12  # Angstrom, the width a shell is broadened to


def labels() -> tuple:
    """The total correlation function followed by every pair of elements.

    VASP names the pairs after the ion types of the structure and reports each of them
    once, so Sr~Ti covers the strontium neighbours of titanium as well.
    """
    pairs = itertools.combinations_with_replacement(_ion_types(), 2)
    return ("total", *(f"{first}~{second}" for first, second in pairs))


def _ion_types():
    """The elements of the crystal in the order the stoichiometry lists them.

    VASP pads the names to a fixed width and stores them as byte strings, so they have
    to be decoded and stripped before they can be compared to an element name.
    """
    stoichiometry = _demo.stoichiometry.Sr2TiO4()
    return [
        convert.text_to_string(ion_type).strip()
        for ion_type in np.array(stoichiometry.ion_types)
    ]


def _elements():
    """The element of every atom of the crystal."""
    stoichiometry = _demo.stoichiometry.Sr2TiO4()
    return np.repeat(_ion_types(), np.array(stoichiometry.number_ion_types))


def Sr2TiO4() -> raw.PairCorrelation:
    """Pair-correlation function of Sr2TiO4 over the steps of the showcase relaxation.

    Rather than a model of a curve, this counts the neighbours of the crystal the
    showcase structure describes at each step, so the first peak sits at the Ti-O bond
    length and the shells migrate outward as the cell expands onto its relaxed size. The
    functions are normalized the way a pair-correlation function is: the integral of
    ``4 pi r^2 rho g(r)`` up to a radius is the number of neighbours within it, so every
    curve settles around one where the shells become dense.
    """
    distances, function = _correlation_functions()
    return raw.PairCorrelation(
        distances=_demo.wrap_data(distances),
        function=_demo.wrap_data(function),
        labels=labels(),
    )


@functools.lru_cache(maxsize=1)
def _correlation_functions():
    """Distances and every correlation function of every step, evaluated once.

    Counting the neighbour shells of twelve steps is by far the most expensive thing
    the showcase computes, and every example in the documentation builds the demo data
    again in the same process, so the result is cached. The arrays are read-only and
    :func:`py4vasp._demo.wrap_data` copies them, so no caller can reach the cache.
    """
    positions = np.array(structure.Sr2TiO4().positions)
    lattice_vectors = np.array(cell.Sr2TiO4().lattice_vectors)
    distances = np.linspace(0.0, MAX_DISTANCE, showcase.NUMBER_POINTS)
    elements = _elements()
    function = np.array(
        [
            _step(distances, step_positions, step_vectors, elements, _ion_types())
            for step_positions, step_vectors in zip(positions, lattice_vectors)
        ]
    )
    distances.setflags(write=False)
    function.setflags(write=False)
    return distances, function


def pair_distances(positions, lattice_vectors) -> np.ndarray:
    """Distance of every atom to every atom of every periodic image within reach.

    Returns
    -------
    -
        Shape ``(atom, atom, image)`` in Angstrom. Distances beyond the range the
        correlation function covers are set to infinity, so the caller can ignore them
        without reshaping.
    """
    shifts = _shifts(lattice_vectors)
    difference = positions[:, None, None, :] - positions[None, :, None, :] + shifts
    distances = np.linalg.norm(difference @ lattice_vectors, axis=-1)
    return np.where(distances <= _reach(), distances, np.inf)


def _reach():
    """Distance out to which shells still contribute to the broadened curve."""
    return MAX_DISTANCE + 5 * WIDTH


def _shifts(lattice_vectors):
    """Integer translations covering every image within reach of the central cell.

    The number needed differs per direction and is not the ratio of the reach to the
    length of the lattice vector: the primitive cell of a body-centred lattice is far
    from orthogonal, so what matters is the spacing between the lattice planes. Taking
    the length instead leaves the outermost shells incomplete, and the curve then
    settles below one rather than around it.
    """
    volume = np.abs(np.linalg.det(lattice_vectors))
    areas = np.linalg.norm(
        np.cross(
            np.roll(lattice_vectors, -1, axis=0), np.roll(lattice_vectors, -2, axis=0)
        ),
        axis=1,
    )
    counts = np.ceil(_reach() * areas / volume).astype(int)
    return np.array(list(itertools.product(*[np.arange(-n, n + 1) for n in counts])))


def _step(distances, positions, lattice_vectors, elements, ion_types):
    """Every labelled correlation function of a single step, shape ``(label, point)``.

    The pairs are visited in the order :func:`labels` names them, so that the rows and
    the labels of the result line up.
    """
    pairs = pair_distances(positions, lattice_vectors)
    volume = np.abs(np.linalg.det(lattice_vectors))
    functions = [_correlate(distances, pairs, volume, len(elements), len(elements))]
    for first, second in itertools.combinations_with_replacement(ion_types, 2):
        selected = pairs[np.ix_(elements == first, elements == second)]
        functions.append(
            _correlate(
                distances,
                selected,
                volume,
                np.count_nonzero(elements == first),
                np.count_nonzero(elements == second),
            )
        )
    return np.array(functions)


def _correlate(distances, pairs, volume, number_first, number_second):
    """Broaden the given shells into g(r) normalized to a uniform neighbour density."""
    shells = pairs[np.isfinite(pairs) & (pairs > 1e-8)]
    neighbours_per_distance = showcase.broaden(distances, shells, width=WIDTH)
    # dividing by the neighbours a uniform density would put in the shell at r makes the
    # curve approach one; the origin has no shell around it, so it is set to zero
    density = number_second / volume
    uniform = 4 * np.pi * distances**2 * density * number_first
    return np.divide(
        neighbours_per_distance, uniform, out=np.zeros_like(distances), where=uniform > 0
    )
