# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo import showcase
from py4vasp._demo.showcase import electronic_structure, kpoint

SPIN_DEGENERACY = 2  # electrons per band without spin polarization
ENERGY_MARGIN = 1.5  # eV of empty axis beyond the outermost eigenvalue

# Indices of the lm-resolved orbitals of py4vasp._demo.projector
_P_ORBITALS = (1, 2, 3)  # py pz px
_T2G_ORBITALS = (4, 5, 7)  # dxy dyz dxz

# Where the states of a manifold sit, as (atom indices, orbital indices, weight per
# atom). The atoms of Sr2TiO4 are two Sr, one Ti, two apical O and two equatorial O; the
# apical and equatorial oxygens occupy different Wyckoff positions, so they contribute
# differently. The valence manifold is the bonding O-2p band with a little Sr-4d and
# Ti-3d admixture, the conduction manifold its Ti-3d t2g antibonding counterpart. Each
# manifold sums to one, so the projections add up to the total density of states.
_VALENCE_CHARACTER = (
    ((0, 1), _T2G_ORBITALS, 0.025),  # Sr
    ((2,), _T2G_ORBITALS, 0.05),  # Ti
    ((3, 4), _P_ORBITALS, 0.25),  # O apical
    ((5, 6), _P_ORBITALS, 0.20),  # O equatorial
)
_CONDUCTION_CHARACTER = (
    ((0, 1), _T2G_ORBITALS, 0.05),  # Sr
    ((2,), _T2G_ORBITALS, 0.72),  # Ti
    ((3, 4), _P_ORBITALS, 0.045),  # O apical
    ((5, 6), _P_ORBITALS, 0.045),  # O equatorial
)


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
    number_atoms, number_orbitals = _shape_of_projections()
    projections = np.zeros((number_atoms, number_orbitals, showcase.NUMBER_POINTS))
    manifolds = (
        (slice(None, model.number_valence_bands), _VALENCE_CHARACTER),
        (slice(model.number_valence_bands, None), _CONDUCTION_CHARACTER),
    )
    for bands, character in manifolds:
        manifold_dos = np.sum(per_band[bands], axis=0)
        for atoms, orbitals, weight in character:
            share = weight * manifold_dos / len(orbitals)
            for atom in atoms:
                projections[atom, list(orbitals)] += share
    return projections[np.newaxis]  # a single spin component


def _shape_of_projections():
    projector = _demo.projector.Sr2TiO4(use_orbitals=True)
    number_atoms = np.sum(projector.stoichiometry.number_ion_types)
    return int(number_atoms), len(projector.orbital_types)
