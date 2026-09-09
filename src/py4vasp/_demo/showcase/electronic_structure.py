# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Analytic band model shared by the showcase band structure and density of states.

Both quantities are evaluated from the same ``E_n(k)``, so the gap a user reads off the
density of states is the gap they see in the band structure. The model is a
few-neighbour tight-binding parametrization tuned to the gap and the bandwidths of the
material; it is not a count-exact one-electron spectrum.
"""

import dataclasses

import numpy as np

NUMBER_VALENCE_BANDS = 6
NUMBER_CONDUCTION_BANDS = 2
BAND_GAP = 2.6  # eV

# Translations by one crystal axis, expressed in the primitive basis of the
# body-centred tetragonal cell: a2 + a3 is the x axis, a1 + a3 the y axis and a1 + a2
# the z axis. Writing the hoppings as integer combinations keeps E_n(k) periodic in the
# Brillouin zone for free, whatever the lattice constants are.
TRANSLATIONS = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

# Centre of every band and the amplitude of its hopping in plane and along c. The
# valence bands stand for the O-2p manifold, the conduction bands for the Ti-3d t2g
# states. Sr2TiO4 is a layered material, so hopping along c is an order of magnitude
# weaker than in plane, which makes the bands flat along the c* direction. Valence and
# conduction amplitudes share their sign, which puts the valence band maximum at Gamma
# and the conduction band minimum at P: the gap is indirect, as it is in the real
# material.
_SR2TIO4_BANDS = (
    # centre  in plane  along c
    (-4.2, 0.35, 0.05),
    (-3.5, 0.30, 0.04),
    (-2.8, 0.40, 0.03),
    (-2.0, 0.45, 0.06),
    (-1.2, 0.50, 0.04),
    (-0.6, 0.55, 0.05),
    (3.2, 0.60, 0.08),
    (4.1, 0.50, 0.05),
)


@dataclasses.dataclass(frozen=True)
class Model:
    """Eigenvalues as ``E_n(k) = centre_n + sum_i weight_ni cos(2 pi k . R_i)``."""

    centers: np.ndarray
    "Centre of every band, shape ``(band,)``."
    weights: np.ndarray
    "Hopping amplitude of every band, shape ``(band, translation)``."
    number_valence_bands: int
    "Number of occupied bands; the remaining ones are empty."
    fermi_energy: float
    "Energy in the middle of the gap."

    @property
    def number_bands(self) -> int:
        return len(self.centers)

    @property
    def valence_band_maximum(self) -> float:
        return float(np.max(self._extrema(self._valence, upper=True)))

    @property
    def conduction_band_minimum(self) -> float:
        return float(np.min(self._extrema(self._conduction, upper=False)))

    def evaluate(self, kpoints) -> np.ndarray:
        """Eigenvalues at the given fractional reciprocal coordinates.

        Parameters
        ----------
        kpoints
            Coordinates as fraction of the reciprocal lattice vectors, shape
            ``(kpoint, 3)``.

        Returns
        -------
        -
            Eigenvalues of shape ``(kpoint, band)``, ascending per k point as VASP
            writes them.
        """
        phase = np.cos(2 * np.pi * np.asarray(kpoints) @ TRANSLATIONS.T)
        return np.sort(self.centers + phase @ self.weights.T, axis=-1)

    @property
    def _valence(self):
        return slice(None, self.number_valence_bands)

    @property
    def _conduction(self):
        return slice(self.number_valence_bands, None)

    def _extrema(self, bands, upper):
        # every cosine reaches +-1 somewhere in the zone, so the hopping amplitudes bound
        # the band; the sign of the amplitudes decides at which k point the bound is met
        spread = np.sum(np.abs(self.weights), axis=1)
        return (self.centers + (spread if upper else -spread))[bands]


def Sr2TiO4() -> Model:
    """Band model of Sr2TiO4: a filled O-2p manifold below an empty Ti-3d t2g pair."""
    centers = np.array([band[0] for band in _SR2TIO4_BANDS])
    in_plane = np.array([band[1] for band in _SR2TIO4_BANDS])
    along_c = np.array([band[2] for band in _SR2TIO4_BANDS])
    # the fourfold axis exchanges the two in-plane translations, so they hop equally
    weights = np.stack((in_plane, in_plane, along_c), axis=-1)
    centers = _align_to_gap(centers, weights, NUMBER_VALENCE_BANDS, BAND_GAP)
    return Model(centers, weights, NUMBER_VALENCE_BANDS, BAND_GAP / 2)


def _align_to_gap(centers, weights, number_valence_bands, band_gap):
    # Shift the valence band maximum onto zero and the conduction band minimum onto the
    # gap, so both are exact rather than whatever the hopping amplitudes happen to give.
    # Every cosine reaches +-1 somewhere in the zone, so the amplitudes bound the bands.
    spread = np.sum(np.abs(weights), axis=1)
    valence = slice(None, number_valence_bands)
    conduction = slice(number_valence_bands, None)
    centers = centers - np.max((centers + spread)[valence])
    centers[conduction] += band_gap - np.min((centers - spread)[conduction])
    return centers
