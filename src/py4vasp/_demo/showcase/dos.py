# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo import showcase
from py4vasp._demo.showcase import electronic_structure, kpoint

SPIN_DEGENERACY = 2  # electrons per band without spin polarization
ENERGY_MARGIN = 1.5  # eV of empty axis beyond the outermost eigenvalue
METAL_BROADENING = 0.35  # eV, wider than the default; see Cu below for why


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
    energies = _energies([model])
    per_band = _broadened_bands(model, energies)
    raw_dos = raw.Dos(
        fermi_energy=model.fermi_energy,
        energies=_demo.wrap_data(energies),
        dos=_demo.wrap_data([np.sum(per_band, axis=0)]),
        projectors=_demo.projector.Sr2TiO4(use_orbitals),
    )
    if use_orbitals:
        raw_dos.projections = _demo.wrap_data(_projections(model, per_band))
    return raw_dos


def _energies(models):
    """Energy axis that covers the eigenvalues of every spin channel with a margin."""
    eigenvalues = [model.evaluate(kpoint.mesh()) for model in models]
    lowest = np.min([np.min(values) for values in eigenvalues]) - ENERGY_MARGIN
    highest = np.max([np.max(values) for values in eigenvalues]) + ENERGY_MARGIN
    return np.linspace(lowest, highest, showcase.NUMBER_POINTS)


def _broadened_bands(model, energies, electrons_per_band=SPIN_DEGENERACY, width=None):
    """Contribution of every band to the density of states, shape ``(band, energy)``."""
    # resolved here rather than as a default argument: this module is imported while the
    # package that holds the constant is still being set up
    width = showcase.BROADENING if width is None else width
    eigenvalues = model.evaluate(kpoint.mesh())
    # dividing by the number of k points turns the sum over the mesh into an average, so
    # every band holds exactly electrons_per_band electrons per unit cell
    weight = electrons_per_band / len(eigenvalues)
    return np.array(
        [
            showcase.broaden(
                energies, eigenvalues[:, band], weights=weight, width=width
            )
            for band in range(model.number_bands)
        ]
    )


def _projections(model, per_band):
    # distribute the contribution of every band over the atoms and orbitals it projects
    # onto; the character sums to one per band, so this adds up to the total again
    character = electronic_structure.Sr2TiO4_character()
    projections = np.einsum("bao,be->aoe", character, per_band)
    return projections[np.newaxis]  # a single spin component


def Fe3O4(projectors, magnetism="collinear"):
    """Spin-resolved density of states of magnetite.

    One channel is gapped at the Fermi energy and the other is not, so the plot shows
    directly that magnetite is a half metal.

    Parameters
    ----------
    projectors
        Pass ``"with_projectors"`` to add the orbital projections.
    magnetism
        Pass ``"noncollinear"`` to resolve the charge and the three spin axes instead of
        the two channels of a collinear calculation.
    """
    models = electronic_structure.Fe3O4()
    use_orbitals = projectors == "with_projectors"
    energies = _energies(models)
    # a spin-polarized calculation resolves the channels, so each band holds one electron
    per_band = [_broadened_bands(model, energies, 1.0) for model in models]
    resolved = _spin_resolved_bands(per_band, magnetism)
    raw_dos = raw.Dos(
        fermi_energy=models[0].fermi_energy,
        energies=_demo.wrap_data(energies),
        dos=_demo.wrap_data(np.sum(resolved, axis=1)),
        projectors=showcase.projector.Fe3O4(use_orbitals, magnetism),
    )
    if use_orbitals:
        character = electronic_structure.Fe3O4_character()
        raw_dos.projections = _demo.wrap_data(
            np.einsum("bao,cbe->caoe", character, resolved)
        )
    return raw_dos


def _spin_resolved_bands(per_band, magnetism):
    """Contribution of every band to every spin component, ``(component, band, energy)``."""
    majority, minority = per_band
    if magnetism == "collinear":
        return np.array([majority, minority])
    # a noncollinear calculation reports the charge and the projection of the
    # magnetization onto the three axes; every band contributes along its own direction
    directions = electronic_structure.Fe3O4_spin_directions()
    projected = np.einsum("ba,be->abe", directions, majority - minority)
    return np.concatenate(([majority + minority], projected))


def Cu(projectors):
    """Density of states of copper, a metal.

    Unlike the other showcase systems, copper carries states at the Fermi energy: the
    narrow d band is filled two to four electronvolts below it and the free-electron band
    crosses it, leaving the low shoulder that makes a metal a metal.
    """
    model = electronic_structure.Cu()
    use_orbitals = projectors == "with_projectors"
    energies = _energies([model])
    # the free-electron band of a metal spans twenty electronvolts, so the k mesh samples
    # it thinly; a metal is anyway usually plotted with more smearing than an insulator
    per_band = _broadened_bands(model, energies, width=METAL_BROADENING)
    raw_dos = raw.Dos(
        fermi_energy=model.fermi_energy,
        energies=_demo.wrap_data(energies),
        dos=_demo.wrap_data([np.sum(per_band, axis=0)]),
        projectors=showcase.projector.Cu(use_orbitals),
    )
    if use_orbitals:
        character = electronic_structure.Cu_character()
        projections = np.einsum("bao,be->aoe", character, per_band)
        raw_dos.projections = _demo.wrap_data(projections[np.newaxis])
    return raw_dos
