# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Band extrema of the showcase crystals over the steps of their trajectory."""
import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo import showcase
from py4vasp._demo.showcase import electronic_structure, kpoint

# The labels VASP writes, in the order it writes them. The k coordinates belong to the
# three extrema above them: the valence band maximum, the conduction band minimum, and
# the k point where the direct gap is smallest.
LABELS = (
    "valence band maximum",
    "conduction band minimum",
    "direct gap bottom",
    "direct gap top",
    "Fermi energy",
    "kx (VBM)",
    "ky (VBM)",
    "kz (VBM)",
    "kx (CBM)",
    "ky (CBM)",
    "kz (CBM)",
    "kx (direct)",
    "ky (direct)",
    "kz (direct)",
)
# How far the band edges of the first step sit from the ones they relax onto. The
# distortion of the first step pushes the oxygen p states up and the titanium d states
# down, so the gap of the distorted crystal is half an electronvolt narrower and opens
# as the relaxation proceeds.
VALENCE_SHIFT = 0.30  # eV
CONDUCTION_SHIFT = -0.20  # eV


def Sr2TiO4() -> raw.Bandgap:
    """Band extrema of Sr2TiO4 over the steps of the showcase relaxation.

    The extrema are read off the same model the showcase band structure and density of
    states are built from, evaluated on a mesh that contains both of them, so the gap
    reported here is the gap those two plots show. It is indirect: the valence band
    maximum sits at Gamma and the conduction band minimum at P.
    """
    model = electronic_structure.Sr2TiO4()
    relaxed = _extrema(model)
    return raw.Bandgap(
        labels=np.array(LABELS, dtype="S"),
        # one spin component: a calculation without spin polarization reports the
        # extrema once rather than for each channel and ignoring the spin
        values=_demo.wrap_data(_trajectory(relaxed)[:, np.newaxis]),
    )


def _extrema(model):
    """Band edges of the relaxed crystal, in the order of :data:`LABELS`."""
    kpoints = kpoint.mesh()
    eigenvalues = model.evaluate(kpoints)
    valence = eigenvalues[:, : model.number_valence_bands]
    conduction = eigenvalues[:, model.number_valence_bands :]
    # the gap at each k point, smallest where the direct transition costs the least
    direct = np.argmin(np.min(conduction, axis=1) - np.max(valence, axis=1))
    return {
        "valence band maximum": np.max(valence),
        "conduction band minimum": np.min(conduction),
        "direct gap bottom": np.max(valence[direct]),
        "direct gap top": np.min(conduction[direct]),
        "Fermi energy": model.fermi_energy,
        "VBM": kpoints[np.argmax(np.max(valence, axis=1))],
        "CBM": kpoints[np.argmin(np.min(conduction, axis=1))],
        "direct": kpoints[direct],
    }


def _trajectory(relaxed):
    """Every labelled value at every step, shape ``(step, label)``."""
    remaining = showcase.decay()
    shifts = {
        "valence band maximum": VALENCE_SHIFT,
        "conduction band minimum": CONDUCTION_SHIFT,
        "direct gap bottom": VALENCE_SHIFT,
        "direct gap top": CONDUCTION_SHIFT,
        "Fermi energy": 0.0,
    }
    energies = [
        relaxed[label] + shift * remaining for label, shift in shifts.items()
    ]
    # the extrema stay at their high-symmetry points while the crystal relaxes, so the
    # k coordinates are the same at every step
    coordinates = [
        np.full_like(remaining, relaxed[name][axis])
        for name in ("VBM", "CBM", "direct")
        for axis in range(3)
    ]
    return np.transpose(energies + coordinates)
