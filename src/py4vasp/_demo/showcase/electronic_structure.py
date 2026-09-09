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

# In a cubic crystal the three primitive translations are equivalent under the
# threefold axis, so a band hops along them with one amplitude.
CUBIC_TRANSLATIONS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# shape of the projections of Sr2TiO4: seven atoms and the lm-resolved orbitals
_SR2TIO4_PROJECTIONS = (7, 16)

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


@dataclasses.dataclass(frozen=True)
class Model:
    """Eigenvalues as ``E_n(k) = centre_n + sum_i weight_ni cos(2 pi k . R_i)``."""

    centers: np.ndarray
    "Centre of every band, shape ``(band,)``."
    weights: np.ndarray
    "Hopping amplitude of every band, shape ``(band, translation)``."
    translations: np.ndarray
    "Lattice translations the model hops along, shape ``(translation, 3)``."
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
        phase = np.cos(2 * np.pi * np.asarray(kpoints) @ self.translations.T)
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
    return Model(centers, weights, TRANSLATIONS, NUMBER_VALENCE_BANDS, BAND_GAP / 2)


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


def Sr2TiO4_character() -> np.ndarray:
    """Weight of every atom and orbital in every band of Sr2TiO4.

    Returns
    -------
    -
        Array of shape ``(band, atom, orbital)`` that sums to one over atoms and
        orbitals, so a quantity distributed with it is fully accounted for. The band
        structure and the density of states share this character, which is why a fat
        band and a projected density of states tell the same story.
    """
    number_atoms, number_orbitals = _SR2TIO4_PROJECTIONS
    character = np.zeros(
        (NUMBER_VALENCE_BANDS + NUMBER_CONDUCTION_BANDS, number_atoms, number_orbitals)
    )
    manifolds = (
        (slice(None, NUMBER_VALENCE_BANDS), _VALENCE_CHARACTER),
        (slice(NUMBER_VALENCE_BANDS, None), _CONDUCTION_CHARACTER),
    )
    for bands, manifold in manifolds:
        for atoms, orbitals, weight in manifold:
            for atom in atoms:
                character[bands, atom, list(orbitals)] = weight / len(orbitals)
    return character


# Magnetite is a ferrimagnetic half metal: the two spin channels see different
# potentials, so one of them opens a gap at the Fermi energy while the other keeps a
# partly filled band there. Centre and hopping amplitude of every band in eV, relative to
# the Fermi energy. The lower bands stand for the O-2p manifold, the upper ones for the
# Fe-3d states the two sublattices contribute.
# A band spans its centre plus or minus the sum of its amplitudes, so with three cubic
# translations its width is six times the amplitude. The majority bands are placed to
# leave a two-electronvolt gap around the Fermi energy, wide enough that the broadening
# of the density of states does not fill it in.
_FE3O4_MAJORITY_BANDS = (
    (-6.5, 0.30),
    (-5.0, 0.35),
    (-3.5, 0.40),
    (-2.0, 0.33),  # highest occupied, reaching up to -1.0
    (2.0, 0.33),  # lowest empty, reaching down to +1.0
    (3.6, 0.30),
)
_FE3O4_MINORITY_BANDS = (
    (-6.2, 0.30),
    (-4.6, 0.35),
    (-3.0, 0.40),
    (-0.2, 0.55),  # the band that crosses the Fermi energy and makes it a half metal
    (2.6, 0.35),
    (4.0, 0.30),
)

# Where the states of magnetite sit. Iron carries the 3d states that make the magnetism,
# oxygen the 2p manifold below them. Both iron sublattices contribute equally to the
# density of states even though their moments point in opposite directions.
_FE3O4_CHARACTER = (
    (range(0, 2), (2,), 0.11),  # Fe tetrahedral, d
    (range(2, 6), (2,), 0.11),  # Fe octahedral, d
    (range(6, 14), (1,), 0.0425),  # O, p
)
_FE3O4_PROJECTIONS = (14, 4)

# Direction the spin of every band points in, as polar and azimuthal angle in degrees.
# The bands the tetrahedral iron dominates point opposite to those the octahedral iron
# dominates, which is the ferrimagnetic order seen with three spin axes instead of two,
# and every direction is canted away from z so that all three components are visible.
_FE3O4_SPIN_DIRECTIONS = (
    (20.0, 35.0),
    (25.0, 40.0),
    (30.0, 45.0),
    (18.0, 30.0),  # the band at the Fermi energy, on the octahedral sublattice
    (160.0, 210.0),  # the tetrahedral sublattice, pointing the other way
    (155.0, 200.0),
)


def Fe3O4() -> list:
    """Band models of the majority and the minority spin channel of magnetite.

    Returns
    -------
    -
        One model per spin channel, both with the Fermi energy at zero. The majority
        channel is gapped there and the minority channel is not, which is what a half
        metal looks like.
    """
    return [
        _cubic_model(_FE3O4_MAJORITY_BANDS, number_valence_bands=4),
        _cubic_model(_FE3O4_MINORITY_BANDS, number_valence_bands=4),
    ]


def _cubic_model(bands, number_valence_bands):
    centers = np.array([band[0] for band in bands])
    amplitude = np.array([band[1] for band in bands])
    weights = np.repeat(amplitude[:, np.newaxis], len(CUBIC_TRANSLATIONS), axis=1)
    return Model(centers, weights, CUBIC_TRANSLATIONS, number_valence_bands, 0.0)


def Fe3O4_character() -> np.ndarray:
    """Weight of every atom and orbital in every band of magnetite.

    Returns
    -------
    -
        Array of shape ``(band, atom, orbital)`` summing to one over atoms and orbitals,
        shared by the spin-resolved band structure and density of states.
    """
    number_bands = len(_FE3O4_MAJORITY_BANDS)
    character = np.zeros((number_bands, *_FE3O4_PROJECTIONS))
    for atoms, orbitals, weight in _FE3O4_CHARACTER:
        for atom in atoms:
            character[:, atom, list(orbitals)] = weight / len(orbitals)
    return character


def Fe3O4_spin_directions() -> np.ndarray:
    """Unit vector the spin of every band points along, shape ``(band, axis)``.

    A collinear calculation only distinguishes parallel from antiparallel; a noncollinear
    one resolves the direction, so the antiparallel iron sublattices of magnetite appear
    as directions that differ by nearly 180 degrees rather than by a sign.
    """
    polar, azimuthal = np.radians(np.array(_FE3O4_SPIN_DIRECTIONS)).T
    return np.array(
        [
            np.sin(polar) * np.cos(azimuthal),
            np.sin(polar) * np.sin(azimuthal),
            np.cos(polar),
        ]
    ).T
