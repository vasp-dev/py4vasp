# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import exception, raw
from py4vasp._demo import (
    CONTCAR,
    band,
    bandgap,
    born_effective_charge,
    cell,
    current_density,
    density,
    dielectric_function,
    dielectric_tensor,
    dispersion,
    dos,
    effective_coulomb,
    elastic_modulus,
    electron_phonon,
    electronic_minimization,
    energy,
    exciton,
    force,
    force_constant,
    internal_strain,
    kpoint,
    local_moment,
    nics,
    pair_correlation,
    partial_density,
    phonon,
    piezoelectric_tensor,
    polarization,
    potential,
    projector,
    stoichiometry,
    stress,
    structure,
    velocity,
    workfunction,
)

# constants for vector dimensions
AXES = 3
# constants for the shape of demo data
NUMBER_ATOMS = 7
NUMBER_BANDS = 3
NUMBER_CHEMICAL_POTENTIALS = 3
NUMBER_CONDUCTION_BANDS = 1
NUMBER_EIGENVECTORS = 5
NUMBER_EXCITONS = 3
NUMBER_FREQUENCIES = 1
NUMBER_MODES = NUMBER_ATOMS * AXES
NUMBER_OMEGA = 12
NUMBER_POINTS = 50
NUMBER_SAMPLES = 5
NUMBER_STEPS = 4
NUMBER_TEMPERATURES = 6
NUMBER_VALENCE_BANDS = 2
NUMBER_WANNIER = 5
# constants for FFT grid dimensions
GRID_DIMENSIONS = (14, 12, 10)  # note: order is z, y, x
# constants for the magnetic configuration
NONPOLARIZED = 1
COLLINEAR = 2
NONCOLLINEAR = 4
# constants to indicate complex data
COMPLEX = 2


def wrap_data(data):
    return raw.VaspData(np.array(data))


def wrap_random_data(shape, present=True, seed=None):
    if present:
        rng = np.random.default_rng(seed)
        data = 10 * rng.standard_normal(shape)
        return raw.VaspData(data)
    else:
        return raw.VaspData(None)


def wrap_orbital_types(use_orbitals, orbital_types):
    if use_orbitals:
        return raw.VaspData(np.array(orbital_types.split(), dtype="S"))
    else:
        return raw.VaspData(None)


def number_components(selection):
    if selection == "collinear":
        return 2
    elif selection in ("noncollinear", "orbital_moments"):
        return 4
    elif selection == "charge_only":
        return 1
    else:
        raise exception.NotImplemented()
