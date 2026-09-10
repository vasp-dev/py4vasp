# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo import showcase
from py4vasp._demo.showcase import electronic_structure, kpoint


def Sr2TiO4(projectors, labels="with_labels"):
    """Band structure of Sr2TiO4 along the high-symmetry path of its Brillouin zone.

    The eigenvalues come from the same model as the showcase density of states, so both
    show the same indirect 2.6 eV gap, and the Fermi energy the two write to the shared
    field of the file agrees.

    Parameters
    ----------
    projectors
        Pass ``"with_projectors"`` to add the orbital projections of a fat band.
    labels
        Pass ``"no_labels"`` to omit the labels of the high-symmetry points, so that
        py4vasp falls back to naming the band edges by their coordinates.
    """
    model = electronic_structure.Sr2TiO4()
    kpoints = kpoint.line_mode(labels)
    eigenvalues, order = electronic_structure.sort_bands(
        model.evaluate(np.array(kpoints.coordinates))
    )
    use_orbitals = projectors == "with_projectors"
    raw_band = raw.Band(
        dispersion=raw.Dispersion(kpoints, _demo.wrap_data([eigenvalues])),
        fermi_energy=model.fermi_energy,
        # the gap is wide, so no state sits close enough to the Fermi energy for the
        # smearing of a real calculation to give it a fractional occupation
        occupations=_demo.wrap_data(
            [np.where(eigenvalues < model.fermi_energy, 1.0, 0.0)]
        ),
        projectors=_demo.projector.Sr2TiO4(use_orbitals),
    )
    if use_orbitals:
        character = electronic_structure.Sr2TiO4_character()
        raw_band.projections = _demo.wrap_data([_project(character, order)])
    return raw_band


def Fe3O4(projectors, labels="with_labels", magnetism="collinear"):
    """Spin-polarized band structure of magnetite along the face-centred cubic path.

    The majority channel is gapped at the Fermi energy while a minority band crosses it,
    so its occupations are fractional across the path: magnetite is a half metal.

    Parameters
    ----------
    projectors
        Pass ``"with_projectors"`` to add the orbital projections.
    labels
        Pass ``"no_labels"`` to leave the high-symmetry points unnamed.
    magnetism
        Pass ``"noncollinear"`` for a single set of eigenvalues that holds the bands of
        both channels, with projections resolving the charge and the three spin axes.
    """
    models = electronic_structure.Fe3O4()
    kpoints = kpoint.line_mode_Fe3O4(labels)
    coordinates = np.array(kpoints.coordinates)
    fermi_energy = models[0].fermi_energy
    use_orbitals = projectors == "with_projectors"
    if magnetism == "collinear":
        eigenvalues, orders = _collinear_bands(models, coordinates)
    else:
        eigenvalues, orders = _noncollinear_bands(models, coordinates)
    raw_band = raw.Band(
        dispersion=raw.Dispersion(kpoints, _demo.wrap_data(eigenvalues)),
        fermi_energy=fermi_energy,
        occupations=_demo.wrap_data(np.where(eigenvalues < fermi_energy, 1.0, 0.0)),
        projectors=showcase.projector.Fe3O4(use_orbitals, magnetism),
    )
    if use_orbitals:
        raw_band.projections = _demo.wrap_data(_Fe3O4_projections(orders, magnetism))
    return raw_band


def Cu(projectors, labels="with_labels"):
    """Band structure of copper along the face-centred cubic path.

    The free-electron band crosses the Fermi energy, so its occupations change along the
    path where the other showcase systems keep theirs at zero or one.
    """
    model = electronic_structure.Cu()
    kpoints = kpoint.line_mode_Cu(labels)
    eigenvalues, order = electronic_structure.sort_bands(
        model.evaluate(np.array(kpoints.coordinates))
    )
    use_orbitals = projectors == "with_projectors"
    raw_band = raw.Band(
        dispersion=raw.Dispersion(kpoints, _demo.wrap_data([eigenvalues])),
        fermi_energy=model.fermi_energy,
        occupations=_demo.wrap_data(
            [np.where(eigenvalues < model.fermi_energy, 1.0, 0.0)]
        ),
        projectors=showcase.projector.Cu(use_orbitals),
    )
    if use_orbitals:
        character = electronic_structure.Cu_character()
        raw_band.projections = _demo.wrap_data([_project(character, order)])
    return raw_band


def _collinear_bands(models, coordinates):
    """One set of eigenvalues per spin channel, each sorted on its own."""
    sorted_channels = [
        electronic_structure.sort_bands(model.evaluate(coordinates)) for model in models
    ]
    eigenvalues = np.array([channel for channel, _ in sorted_channels])
    orders = [order for _, order in sorted_channels]
    return eigenvalues, orders


def _noncollinear_bands(models, coordinates):
    """A single set of eigenvalues holding the bands of both spin channels.

    A noncollinear calculation does not split the bands by spin, but it does not lose
    half of them either: the spinor basis carries as many bands as the two channels have
    together. Taking only one channel would show the gap of that channel while the
    density of states, which sums both, shows states at the Fermi energy.
    """
    channels = [model.evaluate(coordinates) for model in models]
    eigenvalues, order = electronic_structure.sort_bands(
        np.concatenate(channels, axis=-1)
    )
    return eigenvalues[np.newaxis], [order]


def _project(character, order):
    """Project every state with the character of the band it belongs to.

    The character is the one the density of states is decomposed with, so a fat band and
    a projected density of states tell the same story. It follows the permutation that
    sorted the eigenvalues, because the band sitting in a given column changes from one
    k point to the next wherever two bands cross.
    """
    per_state = character[order]  # (kpoint, band, atom, orbital)
    return np.einsum("kbao->aokb", per_state)


def _Fe3O4_projections(orders, magnetism):
    character = electronic_structure.Fe3O4_character()
    if magnetism == "collinear":
        return np.array([_project(character, order) for order in orders])
    # the charge, then the projection of every state onto each of the three spin axes.
    # The bands of the two channels are merged, so the character and the spin direction
    # of both are stacked and follow the same permutation.
    (order,) = orders
    both_channels = np.concatenate(2 * [character])
    directions = electronic_structure.Fe3O4_spin_directions()
    both_directions = np.concatenate((directions, -directions))
    charge = _project(both_channels, order)
    projected = np.einsum("aokb,kbs->saokb", charge, both_directions[order])
    return np.concatenate(([charge], projected))
