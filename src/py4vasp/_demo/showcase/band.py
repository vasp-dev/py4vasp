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
    eigenvalues = model.evaluate(np.array(kpoints.coordinates))
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
        raw_band.projections = _demo.wrap_data(_projections(len(eigenvalues)))
    return raw_band


def _projections(number_kpoints):
    # Every state projects with the character of its band, the same character the
    # density of states is decomposed with. It does not vary along the path, so a fat
    # band has a constant width per band and the contrast is between the bands.
    character = electronic_structure.Sr2TiO4_character()
    constant_along_path = np.ones(number_kpoints)
    projections = np.einsum("bao,k->aokb", character, constant_along_path)
    return projections[np.newaxis]  # a single spin component


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
        Pass ``"noncollinear"`` for a single set of eigenvalues whose projections
        resolve the charge and the three spin axes.
    """
    models = electronic_structure.Fe3O4()
    kpoints = kpoint.line_mode_Fe3O4(labels)
    coordinates = np.array(kpoints.coordinates)
    fermi_energy = models[0].fermi_energy
    use_orbitals = projectors == "with_projectors"
    eigenvalues = _eigenvalues(models, coordinates, magnetism)
    raw_band = raw.Band(
        dispersion=raw.Dispersion(kpoints, _demo.wrap_data(eigenvalues)),
        fermi_energy=fermi_energy,
        occupations=_demo.wrap_data(np.where(eigenvalues < fermi_energy, 1.0, 0.0)),
        projectors=showcase.projector.Fe3O4(use_orbitals, magnetism),
    )
    if use_orbitals:
        raw_band.projections = _demo.wrap_data(
            _Fe3O4_projections(len(coordinates), magnetism)
        )
    return raw_band


def _eigenvalues(models, coordinates, magnetism):
    if magnetism == "collinear":
        return np.array([model.evaluate(coordinates) for model in models])
    # a noncollinear calculation does not split the bands by spin, so it has one set of
    # eigenvalues; the spin shows up in the components of the projections instead
    return np.array([models[0].evaluate(coordinates)])


def _Fe3O4_projections(number_kpoints, magnetism):
    character = electronic_structure.Fe3O4_character()
    constant_along_path = np.ones(number_kpoints)
    projections = np.einsum("bao,k->aokb", character, constant_along_path)
    if magnetism == "collinear":
        return np.array(2 * [projections])
    # the charge, then the projection of every band onto each of the three spin axes
    directions = electronic_structure.Fe3O4_spin_directions()
    projected = np.einsum("aokb,bs->saokb", projections, directions)
    return np.concatenate(([projections], projected))


def Cu(projectors, labels="with_labels"):
    """Band structure of copper along the face-centred cubic path.

    The free-electron band crosses the Fermi energy, so its occupations change along the
    path where the other showcase systems keep theirs at zero or one.
    """
    model = electronic_structure.Cu()
    kpoints = kpoint.line_mode_Cu(labels)
    coordinates = np.array(kpoints.coordinates)
    eigenvalues = model.evaluate(coordinates)
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
        constant_along_path = np.ones(len(coordinates))
        projections = np.einsum("bao,k->aokb", character, constant_along_path)
        raw_band.projections = _demo.wrap_data(projections[np.newaxis])
    return raw_band
