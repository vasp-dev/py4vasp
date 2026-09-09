# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw
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
