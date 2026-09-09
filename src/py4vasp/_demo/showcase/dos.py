# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo import showcase
from py4vasp._demo.showcase import electronic_structure, kpoint

SPIN_DEGENERACY = 2  # electrons per band without spin polarization
ENERGY_MARGIN = 1.5  # eV of empty axis beyond the outermost eigenvalue


def Sr2TiO4(projectors):
    """Density of states of Sr2TiO4 broadened from the band model.

    The eigenvalues are the ones of :func:`electronic_structure.Sr2TiO4` sampled on the
    k mesh, so the gap seen here is the gap of the showcase band structure. Unlike a
    real calculation, where projections only capture the charge inside the PAW spheres,
    the projections sum exactly to the total -- a projected plot that does not add up
    looks like a bug rather than like physics.
    """
    model = electronic_structure.Sr2TiO4()
    use_orbitals = projectors == "with_projectors"
    per_band = _broadened_bands(model)
    raw_dos = raw.Dos(
        fermi_energy=model.fermi_energy,
        energies=_demo.wrap_data(_energies(model)),
        dos=_demo.wrap_data([np.sum(per_band, axis=0)]),
        projectors=_demo.projector.Sr2TiO4(use_orbitals),
    )
    if use_orbitals:
        raw_dos.projections = _demo.wrap_data(_projections(model, per_band))
    return raw_dos


def _energies(model):
    eigenvalues = model.evaluate(kpoint.mesh())
    lowest = np.min(eigenvalues) - ENERGY_MARGIN
    highest = np.max(eigenvalues) + ENERGY_MARGIN
    return np.linspace(lowest, highest, showcase.NUMBER_POINTS)


def _broadened_bands(model):
    """Contribution of every band to the density of states, shape ``(band, energy)``."""
    eigenvalues = model.evaluate(kpoint.mesh())
    # dividing by the number of k points turns the sum over the mesh into an average, so
    # every band holds exactly SPIN_DEGENERACY electrons per unit cell
    weight = SPIN_DEGENERACY / len(eigenvalues)
    energies = _energies(model)
    return np.array(
        [
            showcase.broaden(energies, eigenvalues[:, band], weights=weight)
            for band in range(model.number_bands)
        ]
    )


def _projections(model, per_band):
    # distribute the contribution of every band over the atoms and orbitals it projects
    # onto; the character sums to one per band, so this adds up to the total again
    character = electronic_structure.Sr2TiO4_character()
    projections = np.einsum("bao,be->aoe", character, per_band)
    return projections[np.newaxis]  # a single spin component
