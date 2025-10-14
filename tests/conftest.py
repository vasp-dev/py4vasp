# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import importlib.metadata
import random

import numpy as np
import pytest
from numpy.testing import assert_allclose

from py4vasp import _demo, exception, raw
from py4vasp._util import check, import_

stats = import_.optional("scipy.stats")

number_steps = 4
number_atoms = 7
number_points = 50
number_bands = 3
number_valence_bands = 2
number_conduction_bands = 1
number_eigenvectors = 5
number_excitons = 3
number_samples = 5
number_chemical_potentials = 3
number_temperatures = 6
number_frequencies = (
    1  # number of frequencies at which the fan self-energy is evaluated
)
single_spin = 1
two_spins = 2
noncollinear = 4
axes = 3
complex_ = 2
number_modes = axes * number_atoms
grid_dimensions = (14, 12, 10)  # note: order is z, y, x


@pytest.fixture(scope="session")
def only_core():
    if not _is_core():
        pytest.skip("This test checks py4vasp-core functionality not used by py4vasp.")


@pytest.fixture(scope="session")
def not_core():
    if _is_core():
        pytest.skip("This test requires features not present in py4vasp-core.")


def _is_core():
    try:
        importlib.metadata.distribution("py4vasp-core")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


class _Assert:
    @staticmethod
    def allclose(actual, desired, tolerance=1):
        if check.is_none(actual):
            assert check.is_none(desired)
        elif dataclasses.is_dataclass(actual):
            _compare_dataclasses(actual, desired, tolerance)
        else:
            _compare_arrays(actual, desired, tolerance)

    @staticmethod
    def same_structure(actual, desired):
        for key in actual:
            if key in ("elements", "names"):
                assert actual[key] == desired[key]
            else:
                _Assert.allclose(actual[key], desired[key])

    @staticmethod
    def same_structure_view(actual, desired):
        assert np.all(actual.elements == desired.elements)
        _Assert.allclose(actual.lattice_vectors, desired.lattice_vectors)
        _Assert.allclose(actual.positions, desired.positions)
        _Assert.allclose(actual.supercell, desired.supercell)

    @staticmethod
    def same_raw_contcar(actual, desired, exact_match=True):
        # exact_match requires cell to be identical and not just equivalent
        _Assert.same_raw_structure(actual.structure, desired.structure, exact_match)
        assert actual.system == desired.system
        _Assert.allclose(actual.selective_dynamics, desired.selective_dynamics)
        # velocities use a lower precision when writing
        _Assert.allclose(
            actual.lattice_velocities.astype(np.float32),
            desired.lattice_velocities.astype(np.float32),
        )
        _Assert.allclose(
            actual.ion_velocities.astype(np.float32),
            desired.ion_velocities.astype(np.float32),
        )

    @staticmethod
    def same_raw_structure(actual, desired, exact_match=True):
        # exact_match requires cell to be identical and not just equivalent
        _Assert.allclose(actual.stoichiometry, desired.stoichiometry)
        _Assert.same_raw_cell(actual.cell, desired.cell, exact_match)
        _Assert.allclose(actual.positions, desired.positions)

    @staticmethod
    def same_raw_cell(actual, desired, exact_match=True):
        # exact_match requires cell to be identical and not just equivalent
        if exact_match:
            _Assert.allclose(actual.lattice_vectors, desired.lattice_vectors)
            _Assert.allclose(actual.scale, desired.scale)
        else:
            actual_lattice_vectors = actual.lattice_vectors * actual.scale
            desired_lattice_vectors = desired.lattice_vectors * desired.scale
            _Assert.allclose(actual_lattice_vectors, desired_lattice_vectors)


def _compare_dataclasses(actual, desired, tolerance):
    assert dataclasses.is_dataclass(desired)
    assert type(actual) is type(desired)
    for field in dataclasses.fields(actual):
        actual_field = getattr(actual, field.name)
        desired_field = getattr(desired, field.name)
        _Assert.allclose(actual_field, desired_field, tolerance)


def _compare_arrays(actual, desired, tolerance):
    actual = np.array(actual)
    desired = np.array(desired)
    type_ = actual.dtype.type
    if type_ in (np.bool_, np.str_, np.bytes_):
        assert type_ == desired.dtype.type
        assert np.array_equal(actual, desired)
    else:
        actual, desired = np.broadcast_arrays(actual, desired)
        actual, mask_actual = _finite_subset(actual)
        desired, mask_desired = _finite_subset(desired)
        assert np.all(mask_actual == mask_desired)
        tolerance = 1e-14 * tolerance
        assert_allclose(actual, desired, rtol=tolerance, atol=tolerance)


def _finite_subset(array):
    array = np.atleast_1d(array)
    mask = np.isfinite(array)
    return array[mask], mask


@pytest.fixture(scope="session")
def Assert():
    return _Assert


class RawDataFactory:
    @staticmethod
    def band(selection):
        band, *options = selection.split()
        options = options[0] if len(options) > 0 else None
        if band == "single":
            return _demo.band.single_band()
        elif band == "multiple":
            return _demo.band.multiple_bands(options)
        elif band == "line":
            return _demo.band.line_mode(options)
        elif band == "spin_polarized":
            return _demo.band.spin_polarized_bands(options)
        elif band == "noncollinear":
            return _demo.band.noncollinear_bands(options)
        elif band == "spin_texture":
            return _demo.band.spin_texture(options)
        else:
            raise exception.NotImplemented()

    @staticmethod
    def bandgap(selection):
        return _demo.bandgap.bandgap(selection)

    @staticmethod
    def born_effective_charge(selection):
        if selection == "Sr2TiO4":
            return _demo.born_effective_charge.Sr2TiO4()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def CONTCAR(selection):
        if selection == "Sr2TiO4":
            return _demo.CONTCAR.Sr2TiO4()
        elif selection == "Fe3O4":
            return _demo.CONTCAR.Fe3O4()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def current_density(selection):
        return _demo.current_density.current_density(selection)

    @staticmethod
    def density(selection):
        parts = selection.split()
        if parts[0] == "Sr2TiO4":
            return _demo.density.Sr2TiO4()
        elif parts[0] == "Fe3O4":
            return _demo.density.Fe3O4(parts[1])
        else:
            raise exception.NotImplemented()

    @staticmethod
    def dielectric_function(selection):
        if selection == "electron":
            return _demo.dielectric_function.electron()
        elif selection == "ion":
            return _demo.dielectric_function.ionic()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def dielectric_tensor(selection):
        method, with_ion = selection.split()
        with_ion = with_ion == "with_ion"
        return _demo.dielectric_tensor.dielectric_tensor(method, with_ion)

    @staticmethod
    def dispersion(selection):
        if selection == "single_band":
            return _demo.dispersion.single_band()
        elif selection == "multiple_bands":
            return _demo.dispersion.multiple_bands()
        elif selection == "line":
            return _demo.dispersion.line_mode("no_labels")
        elif selection == "spin_polarized":
            return _demo.dispersion.spin_polarized_bands()
        elif selection == "phonon":
            return _demo.dispersion.phonon()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def dos(selection):
        structure, *projectors = selection.split()
        projectors = projectors[0] if len(projectors) > 0 else "no_projectors"
        if structure == "Sr2TiO4":
            return _demo.dos.Sr2TiO4(projectors)
        elif structure == "Fe3O4":
            return _demo.dos.Fe3O4(projectors)
        elif structure == "Ba2PbO4":
            return _demo.dos.Ba2PbO4(projectors)
        else:
            raise exception.NotImplemented()

    @staticmethod
    def elastic_modulus(selection):
        return _demo.elastic_modulus.elastic_modulus()

    @staticmethod
    def electron_phonon_band_gap(selection):
        return _demo.electron_phonon.bandgap.bandgap(selection)

    @staticmethod
    def electron_phonon_chemical_potential(selection):
        return _demo.electron_phonon.chemical_potential.chemical_potential(selection)

    @staticmethod
    def electron_phonon_self_energy(selection):
        return _demo.electron_phonon.self_energy.self_energy(selection)

    @staticmethod
    def electron_phonon_transport(selection):
        return _demo.electron_phonon.transport.transport(selection)

    @staticmethod
    def electronic_minimization(selection=None):
        return _demo.electronic_minimization.electronic_minimization()

    @staticmethod
    def energy(selection, randomize: bool = False):
        if selection == "MD":
            return _demo.energy.MD(randomize)
        elif selection == "relax":
            return _demo.energy.relax(randomize)
        elif selection == "afqmc":
            return _demo.energy.afqmc()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def exciton_density():
        return _demo.exciton.density.Sr2TiO4()

    @staticmethod
    def exciton_eigenvector(selection):
        return _demo.exciton.eigenvector.Sr2TiO4()

    @staticmethod
    def force_constant(selection):
        if selection == "Sr2TiO4 all atoms":
            return _demo.force_constant.Sr2TiO4(use_selective_dynamics=False)
        if selection == "Sr2TiO4 selective dynamics":
            return _demo.force_constant.Sr2TiO4(use_selective_dynamics=True)
        else:
            raise exception.NotImplemented()

    @staticmethod
    def force(selection, randomize: bool = False):
        if selection == "Sr2TiO4":
            return _demo.force.Sr2TiO4(randomize)
        elif selection == "Fe3O4":
            return _demo.force.Fe3O4(randomize)
        else:
            raise exception.NotImplemented()

    @staticmethod
    def internal_strain(selection):
        if selection == "Sr2TiO4":
            return _demo.internal_strain.Sr2TiO4()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def kpoint(selection):
        if selection == "qpoints":
            return _demo.kpoint.qpoints()
        mode, *labels = selection.split()
        labels = labels[0] if len(labels) > 0 else "no_labels"
        if mode[0] in ["l", b"l"[0]]:
            return _demo.kpoint.line_mode(mode, labels)
        else:
            return _demo.kpoint.grid(mode, labels)

    @staticmethod
    def local_moment(selection):
        return _demo.local_moment.local_moment(selection)

    @staticmethod
    def nics(selection):
        if selection == "on-a-grid":
            return _demo.nics.Sr2TiO4()
        if selection == "at-points":
            return _demo.nics.Fe3O4()

    @staticmethod
    def pair_correlation(selection):
        return _demo.pair_correlation.Sr2TiO4()

    @staticmethod
    def partial_density(selection):
        return _partial_density(selection)

    @staticmethod
    def piezoelectric_tensor(selection):
        return _demo.piezoelectric_tensor.piezoelectric_tensor()

    @staticmethod
    def polarization(selection):
        return _demo.polarization.polarization()

    @staticmethod
    def phonon_band(selection):
        return _demo.phonon.band.Sr2TiO4()

    @staticmethod
    def phonon_dos(selection):
        return _demo.phonon.dos.Sr2TiO4()

    @staticmethod
    def phonon_mode(selection):
        return _demo.phonon.mode.Sr2TiO4()

    @staticmethod
    def potential(selection: str):
        parts = selection.split()
        if parts[0] == "Sr2TiO4":
            return _demo.potential.Sr2TiO4(parts[1])
        elif parts[0] == "Fe3O4":
            return _demo.potential.Fe3O4(*parts[1:])
        else:
            raise exception.NotImplemented()

    @staticmethod
    def projector(selection):
        if selection == "Sr2TiO4":
            return _demo.projector.Sr2TiO4(use_orbitals=True)
        elif selection == "Fe3O4":
            return _demo.projector.Fe3O4(use_orbitals=True)
        elif selection == "Ba2PbO4":
            return _demo.projector.Ba2PbO4(use_orbitals=True)
        elif selection == "without_orbitals":
            return _demo.projector.Sr2TiO4(use_orbitals=False)
        else:
            raise exception.NotImplemented()

    @staticmethod
    def stress(selection, randomize: bool = False):
        if selection == "Sr2TiO4":
            return _Sr2TiO4_stress(randomize)
        elif selection == "Fe3O4":
            return _Fe3O4_stress(randomize)
        else:
            raise exception.NotImplemented()

    @staticmethod
    def structure(selection):
        if selection == "BN":
            return _BN_structure()
        elif selection == "Ca3AsBr3":
            return _Ca3AsBr3_structure()
        elif selection == "Fe3O4":
            return _Fe3O4_structure()
        elif selection == "SrTiO3":
            return _SrTiO3_structure()
        elif selection == "Sr2TiO4":
            return _Sr2TiO4_structure()
        elif selection == "Sr2TiO4 without ion types":
            return _Sr2TiO4_structure(has_ion_types=False)
        elif selection == "ZnS":
            return _ZnS_structure()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def stoichiometry(selection):
        if selection == "Sr2TiO4":
            return _demo.stoichiometry.Sr2TiO4()
        elif selection == "Sr2TiO4 without ion types":
            return _demo.stoichiometry.Sr2TiO4(has_ion_types=False)
        elif selection == "Fe3O4":
            return _demo.stoichiometry.Fe3O4()
        elif selection == "Ca2AsBr-CaBr2":  # test duplicate entries
            return _demo.stoichiometry.Ca3AsBr3()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def velocity(selection):
        if selection == "Sr2TiO4":
            return _demo.velocity.Sr2TiO4()
        elif selection == "Fe3O4":
            return _demo.velocity.Fe3O4()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def workfunction(selection):
        return _demo.workfunction.workfunction(selection)


@pytest.fixture(scope="session")
def raw_data():
    return RawDataFactory


def _partial_density(selection):
    grid_dim = grid_dimensions
    if "CaAs3_110" in selection:
        structure = _CaAs3_110_structure()
        grid_dim = (240, 40, 32)
    elif "Sr2TiO4" in selection:
        structure = _Sr2TiO4_structure()
    elif "Ca3AsBr3" in selection:
        structure = _Ca3AsBr3_structure()
    elif "Ni100" in selection:
        structure = _Ni100_structure()
    else:
        structure = _Graphite_structure()
        grid_dim = (216, 24, 24)
    if "split_bands" in selection:
        bands = raw.VaspData(random.sample(range(1, 51), 3))
    else:
        bands = raw.VaspData(np.asarray([0]))
    if "split_kpoints" in selection:
        kpoints = raw.VaspData((random.sample(range(1, 26), 5)))
    else:
        kpoints = raw.VaspData(np.asarray([0]))
    if "spin_polarized" in selection:
        spin_dimension = 2
    else:
        spin_dimension = 1
    grid = raw.VaspData(tuple(reversed(grid_dim)))
    charge_shape = (len(kpoints), len(bands), spin_dimension, *grid_dim)
    gaussian_charge = np.zeros(charge_shape)
    if not _is_core():
        cov = grid_dim[0] / 10  # standard deviation
        z = np.arange(grid_dim[0])  # z range
        for gy in range(grid_dim[1]):
            for gx in range(grid_dim[2]):
                m = int(grid_dim[0] / 2) + gy / 10 + gx / 10
                val = stats.multivariate_normal(mean=m, cov=cov).pdf(z)
                # Fill the gaussian_charge array
                gaussian_charge[:, :, :, :, gy, gx] = val
    gaussian_charge = raw.VaspData(gaussian_charge)
    return raw.PartialDensity(
        structure=structure,
        bands=bands,
        kpoints=kpoints,
        partial_charge=gaussian_charge,
        grid=grid,
    )


def _Sr2TiO4_stress(randomize):
    shape = (number_steps, axes, axes)
    if randomize:
        stresses = np.random.random(shape)
    else:
        stresses = np.arange(np.prod(shape)).reshape(shape)
    return raw.Stress(structure=_Sr2TiO4_structure(), stress=stresses)


def _Graphite_structure():
    # repetitions = (number_steps, 1, 1)
    positions = [
        [0.00000000, 0.00000000, 0.00000000],
        [0.33333333, 0.66666667, 0.00000000],
        [0.33333333, 0.66666667, 0.15031929],
        [0.66666667, 0.33333333, 0.15031929],
        [0.00000000, 0.00000000, 0.30063858],
        [0.33333333, 0.66666667, 0.30063858],
        [0.33333333, 0.66666667, 0.45095787],
        [0.66666667, 0.33333333, 0.45095787],
        [0.00000000, 0.00000000, 0.60127716],
        [0.33333333, 0.66666667, 0.60127716],
    ]
    return raw.Structure(
        stoichiometry=_Graphite_stoichiometry(),
        cell=_Graphite_cell(),
        positions=raw.VaspData(positions),
    )


def _Graphite_cell():
    lattice_vectors = [
        [2.44104624, 0.00000000, 0.00000000],
        [-1.22052312, 2.11400806, 0.00000000],
        [0.00000000, 0.00000000, 22.0000000],
    ]
    return raw.Cell(np.asarray(lattice_vectors), scale=raw.VaspData(1.0))


def _Graphite_stoichiometry():
    return raw.Stoichiometry(
        number_ion_types=np.array((10,)),
        ion_types=np.array(("C",), dtype="S"),
    )


def _Ni100_structure():
    # repetitions = (number_steps, 1, 1)
    positions = [
        [0.00000000, 0.00000000, 0.00000000],
        [0.50000000, 0.10000000, 0.50000000],
        [0.00000000, 0.20000000, 0.00000000],
        [0.50000000, 0.30000000, 0.50000000],
        [0.00000000, 0.40000000, 0.00000000],
    ]
    return raw.Structure(
        stoichiometry=_Ni100_stoichiometry(),
        cell=_Ni100_cell(),
        positions=raw.VaspData(positions),
    )


def _Ni100_cell():
    lattice_vectors = [
        [2.496086836, 0.00000000, 0.00000000],
        [-1.22052312, 35.2999992371, 0.00000000],
        [0.00000000, 0.00000000, 2.4960868359],
    ]
    return raw.Cell(np.asarray(lattice_vectors), scale=raw.VaspData(1.0))


def _Ni100_stoichiometry():
    return raw.Stoichiometry(
        number_ion_types=np.array((5,)),
        ion_types=np.array(("Ni",), dtype="S"),
    )


def _CaAs3_110_structure():
    # repetitions = (number_steps, 1, 1)
    positions = [
        [0.20000458, 0.51381288, 0.73110298],
        [0.79999542, 0.48618711, 0.66008269],
        [0.20000458, 0.51381288, 0.93991731],
        [0.70000458, 0.01381289, 0.83551014],
        [0.79999542, 0.48618711, 0.86889702],
        [0.29999541, 0.98618712, 0.76448986],
        [0.08920607, 0.11201309, 0.67393241],
        [0.91079393, 0.88798690, 0.71725325],
        [0.57346071, 0.83596581, 0.70010722],
        [0.42653929, 0.16403419, 0.69107845],
        [0.72035614, 0.40406032, 0.73436505],
        [0.27964386, 0.59593968, 0.65682062],
        [0.08920607, 0.11201309, 0.88274675],
        [0.58920607, 0.61201310, 0.77833958],
        [0.91079393, 0.88798690, 0.92606759],
        [0.41079393, 0.38798690, 0.82166042],
        [0.57346071, 0.83596581, 0.90892155],
        [0.07346071, 0.33596581, 0.80451438],
        [0.42653929, 0.16403419, 0.89989278],
        [0.92653929, 0.66403419, 0.79548562],
        [0.72035614, 0.40406032, 0.94317938],
        [0.22035614, 0.90406032, 0.83877221],
        [0.27964386, 0.59593968, 0.86563495],
        [0.77964386, 0.09593968, 0.76122779],
    ]
    return raw.Structure(
        stoichiometry=_CaAs3_110_stoichiometry(),
        cell=_CaAs3_110_cell(),
        positions=raw.VaspData(positions),
    )


def _CaAs3_110_cell():
    lattice_vectors = [
        [5.65019183, 0.00000000, 1.90320681],
        [0.85575829, 7.16802977, 0.65250675],
        [0.00000000, 0.00000000, 44.41010402],
    ]
    return raw.Cell(np.asarray(lattice_vectors), scale=raw.VaspData(1.0))


def _CaAs3_110_stoichiometry():
    return raw.Stoichiometry(
        number_ion_types=np.array((6, 18)),
        ion_types=np.array(("Ca", "As"), dtype="S"),
    )


def _Sr2TiO4_structure(has_ion_types=True):
    return _demo.structure.Sr2TiO4(has_ion_types=has_ion_types)


def _Fe3O4_stress(randomize):
    shape = (number_steps, axes, axes)
    if randomize:
        stresses = np.random.random(shape)
    else:
        stresses = np.arange(np.prod(shape)).reshape(shape)
    return raw.Stress(
        structure=_Fe3O4_structure(),
        stress=stresses,
    )


def _Fe3O4_structure():
    return _demo.structure.Fe3O4()


def _Ca3AsBr3_cell():
    return raw.Cell(
        scale=raw.VaspData(5.93),
        lattice_vectors=_make_data(np.eye(3)),
    )


def _Ca3AsBr3_structure():
    positions = [
        [0.5, 0.0, 0.0],  # Ca_1
        [0.0, 0.5, 0.0],  # Ca_2
        [0.0, 0.0, 0.0],  # As
        [0.0, 0.5, 0.5],  # Br_1
        [0.0, 0.0, 0.5],  # Ca_3
        [0.5, 0.0, 0.5],  # Br_2
        [0.5, 0.5, 0.0],  # Br_3
    ]
    return raw.Structure(
        stoichiometry=_Ca3AsBr3_stoichiometry(),
        cell=_Ca3AsBr3_cell(),
        positions=_make_data(positions),
    )


def _Ca3AsBr3_stoichiometry():
    return _demo.stoichiometry.Ca3AsBr3()


def _ZnS_structure():
    return raw.Structure(
        raw.Stoichiometry(number_ion_types=[2, 2], ion_types=["Zn", "S"]),
        raw.Cell(
            lattice_vectors=np.array([[1.9, -3.3, 0.0], [1.9, 3.3, 0.0], [0, 0, 6.2]]),
            scale=raw.VaspData(1.0),
        ),
        positions=np.array(
            [
                [1 / 3, 2 / 3, 0.0],
                [2 / 3, 1 / 3, 0.5],
                [1 / 3, 2 / 3, 0.375],
                [2 / 3, 1 / 3, 0.875],
            ]
        ),
    )


def _SrTiO3_structure():
    return raw.Structure(
        raw.Stoichiometry(number_ion_types=[1, 1, 3], ion_types=["Sr", "Ti", "O"]),
        raw.Cell(lattice_vectors=np.eye(3), scale=raw.VaspData(4.0)),
        positions=np.array(
            [
                [0, 0, 0],
                [0.5, 0.5, 0.5],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ]
        ),
    )


def _BN_structure():
    return raw.Structure(
        raw.Stoichiometry(number_ion_types=[1, 1], ion_types=["B", "N"]),
        raw.Cell(
            lattice_vectors=np.array(
                [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
            ),
            scale=raw.VaspData(3.63),
        ),
        positions=np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
    )


def _make_data(data):
    return raw.VaspData(np.array(data))
