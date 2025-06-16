# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import importlib.metadata
import itertools
import random

import numpy as np
import pytest
from numpy.testing import assert_allclose

from py4vasp import exception, raw
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
            return _single_band(options)
        elif band == "multiple":
            return _multiple_bands(options)
        elif band == "line":
            return _line_band(options)
        elif band == "spin_polarized":
            return _spin_polarized_bands(options)
        elif band == "noncollinear":
            return _noncollinear_bands(options)
        elif band == "spin_texture":
            return _spin_texture_bands(options)
        elif band == "noncollinear":
            return _noncollinear_bands(options)
        elif band == "spin_texture":
            return _spin_texture_bands(options)
        else:
            raise exception.NotImplemented()

    @staticmethod
    def bandgap(selection):
        return _bandgap(selection)

    @staticmethod
    def born_effective_charge(selection):
        if selection == "Sr2TiO4":
            return _Sr2TiO4_born_effective_charges()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def CONTCAR(selection):
        if selection == "Sr2TiO4":
            return _Sr2TiO4_CONTCAR()
        elif selection == "Fe3O4":
            return _Fe3O4_CONTCAR()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def current_density(selection):
        return _current_density(selection)

    @staticmethod
    def density(selection):
        parts = selection.split()
        if parts[0] == "Sr2TiO4":
            return _Sr2TiO4_density()
        elif parts[0] == "Fe3O4":
            return _Fe3O4_density(parts[1])
        else:
            raise exception.NotImplemented()

    @staticmethod
    def dielectric_function(selection):
        if selection == "electron":
            return _electron_dielectric_function()
        elif selection == "ion":
            return _ion_dielectric_function()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def dielectric_tensor(selection):
        method, with_ion = selection.split()
        with_ion = with_ion == "with_ion"
        return _dielectric_tensor(method, with_ion)

    @staticmethod
    def dispersion(selection):
        if selection == "single_band":
            return _single_band_dispersion()
        elif selection == "multiple_bands":
            return _multiple_bands_dispersion()
        elif selection == "line":
            return _line_dispersion("no_labels")
        elif selection == "spin_polarized":
            return _spin_polarized_dispersion()
        elif selection == "phonon":
            return _phonon_dispersion()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def dos(selection):
        structure, *projectors = selection.split()
        projectors = projectors[0] if len(projectors) > 0 else "no_projectors"
        if structure == "Sr2TiO4":
            return _Sr2TiO4_dos(projectors)
        elif structure == "Fe3O4":
            return _Fe3O4_dos(projectors)
        elif structure == "Ba2PbO4":
            return _Ba2PbO4_dos(projectors)
        else:
            raise exception.NotImplemented()

    @staticmethod
    def elastic_modulus(selection):
        return _elastic_modulus()

    @staticmethod
    def electron_phonon_band_gap(selection):
        return _electron_phonon_band_gap(selection)

    @staticmethod
    def electron_phonon_chemical_potential(selection):
        return _electron_phonon_chemical_potential(selection)

    @staticmethod
    def electron_phonon_self_energy(selection):
        return _electron_phonon_self_energy(selection)

    @staticmethod
    def electron_phonon_transport(selection):
        return _electron_phonon_transport(selection)

    @staticmethod
    def electronic_minimization(selection=None):
        return _electronic_minimization()

    @staticmethod
    def energy(selection, randomize: bool = False):
        if selection == "MD":
            return _MD_energy(randomize)
        elif selection == "relax":
            return _relax_energy(randomize)
        elif selection == "afqmc":
            return _afqmc_energy()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def exciton_density():
        return _Sr2TiO4_exciton_density()

    @staticmethod
    def exciton_eigenvector(selection):
        return _Sr2TiO4_exciton_eigenvector()

    @staticmethod
    def force_constant(selection):
        if selection == "Sr2TiO4 all atoms":
            return _Sr2TiO4_force_constants(use_selective_dynamics=False)
        if selection == "Sr2TiO4 selective dynamics":
            return _Sr2TiO4_force_constants(use_selective_dynamics=True)
        else:
            raise exception.NotImplemented()

    @staticmethod
    def force(selection, randomize: bool = False):
        if selection == "Sr2TiO4":
            return _Sr2TiO4_forces(randomize)
        elif selection == "Fe3O4":
            return _Fe3O4_forces(randomize)
        else:
            raise exception.NotImplemented()

    @staticmethod
    def internal_strain(selection):
        if selection == "Sr2TiO4":
            return _Sr2TiO4_internal_strain()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def kpoint(selection):
        if selection == "qpoints":
            return _qpoints()
        mode, *labels = selection.split()
        labels = labels[0] if len(labels) > 0 else "no_labels"
        if mode[0] in ["l", b"l"[0]]:
            return _line_kpoints(mode, labels)
        else:
            return _grid_kpoints(mode, labels)

    @staticmethod
    def local_moment(selection):
        return _local_moment(selection)

    @staticmethod
    def nics(selection):
        if selection == "on-a-grid":
            return _Sr2TiO4_nics()
        if selection == "at-points":
            return _Fe3O4_nics()

    @staticmethod
    def pair_correlation(selection):
        return _Sr2TiO4_pair_correlation()

    @staticmethod
    def partial_density(selection):
        return _partial_density(selection)

    @staticmethod
    def piezoelectric_tensor(selection):
        return _piezoelectric_tensor()

    @staticmethod
    def polarization(selection):
        return _polarization()

    @staticmethod
    def phonon_band(selection):
        return _phonon_band()

    @staticmethod
    def phonon_dos(selection):
        return _phonon_dos()

    @staticmethod
    def phonon_mode(selection):
        return _phonon_mode()

    @staticmethod
    def potential(selection: str):
        parts = selection.split()
        if parts[0] == "Sr2TiO4":
            return _Sr2TiO4_potential(parts[1])
        elif parts[0] == "Fe3O4":
            return _Fe3O4_potential(*parts[1:])
        else:
            raise exception.NotImplemented()

    @staticmethod
    def projector(selection):
        if selection == "Sr2TiO4":
            return _Sr2TiO4_projectors(use_orbitals=True)
        elif selection == "Fe3O4":
            return _Fe3O4_projectors(use_orbitals=True)
        elif selection == "Ba2PbO4":
            return _Ba2PbO4_projectors(use_orbitals=True)
        elif selection == "without_orbitals":
            return _Sr2TiO4_projectors(use_orbitals=False)
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
            return _Sr2TiO4_stoichiometry()
        elif selection == "Sr2TiO4 without ion types":
            return _Sr2TiO4_stoichiometry(has_ion_types=False)
        elif selection == "Fe3O4":
            return _Fe3O4_stoichiometry()
        elif selection == "Ca2AsBr-CaBr2":  # test duplicate entries
            return _Ca3AsBr3_stoichiometry()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def velocity(selection):
        if selection == "Sr2TiO4":
            return _Sr2TiO4_velocity()
        elif selection == "Fe3O4":
            return _Fe3O4_velocity()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def workfunction(selection):
        return _workfunction(selection)


@pytest.fixture(scope="session")
def raw_data():
    return RawDataFactory


def _number_components(selection):
    if selection == "collinear":
        return 2
    elif selection in ("noncollinear", "orbital_moments"):
        return 4
    elif selection == "charge_only":
        return 1
    else:
        raise exception.NotImplemented()


def _bandgap(selection):
    labels = (
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
    num_components = 3 if selection == "spin_polarized" else 1
    shape = (number_steps, num_components, len(labels))
    data = np.sqrt(np.arange(np.prod(shape)).reshape(shape))
    if num_components == 3:
        # only spin-independent Fermi energy implemented
        data[:, 1, 4] = data[:, 0, 4]
        data[:, 2, 4] = data[:, 0, 4]
    return raw.Bandgap(labels=np.array(labels, dtype="S"), values=data)


def _electron_dielectric_function():
    shape = (2, axes, axes, number_points, complex_)
    data = np.linspace(0, 1, np.prod(shape)).reshape(shape)
    return raw.DielectricFunction(
        energies=np.linspace(0, 1, number_points),
        dielectric_function=_make_data(data[0]),
        current_current=_make_data(data[1]),
    )


def _ion_dielectric_function():
    shape = (axes, axes, number_points, complex_)
    data = np.linspace(0, 1, np.prod(shape)).reshape(shape)
    return raw.DielectricFunction(
        energies=np.linspace(0, 1, number_points),
        dielectric_function=_make_data(data),
        current_current=raw.VaspData(None),
    )


def _dielectric_tensor(method, with_ion):
    shape = (3, axes, axes)
    data = np.arange(np.prod(shape)).reshape(shape)
    ion = raw.VaspData(data[1] if with_ion else None)
    independent_particle = raw.VaspData(data[2] if method in ("dft", "rpa") else None)
    return raw.DielectricTensor(
        electron=raw.VaspData(data[0]),
        ion=ion,
        independent_particle=independent_particle,
        method=method.encode(),
    )


def _elastic_modulus():
    shape = (2, axes, axes, axes, axes)
    data = np.arange(np.prod(shape)).reshape(shape)
    return raw.ElasticModulus(clamped_ion=data[0], relaxed_ion=data[1])


def _phonon_band():
    dispersion = _phonon_dispersion()
    shape = (*dispersion.eigenvalues.shape, number_atoms, axes, complex_)
    return raw.PhononBand(
        dispersion=dispersion,
        stoichiometry=_Sr2TiO4_stoichiometry(),
        eigenvectors=np.linspace(0, 1, np.prod(shape)).reshape(shape),
    )


def _phonon_dispersion():
    qpoints = _qpoints()
    shape = (len(qpoints.coordinates), number_modes)
    eigenvalues = np.arange(np.prod(shape)).reshape(shape)
    return raw.Dispersion(qpoints, eigenvalues)


def _phonon_dos():
    energies = np.linspace(0, 5, number_points)
    dos = energies**2
    lower_ratio = np.arange(number_modes, dtype=np.float64).reshape(axes, number_atoms)
    lower_ratio /= np.sum(lower_ratio)
    upper_ratio = np.array(list(reversed(lower_ratio)))
    ratio = np.linspace(lower_ratio, upper_ratio, number_points).T
    projections = np.multiply(ratio, dos)
    return raw.PhononDos(energies, dos, projections, _Sr2TiO4_stoichiometry())


def _phonon_mode():
    frequencies = np.sqrt(np.linspace(0.1, -0.02, number_modes, dtype=np.complex128))
    return raw.PhononMode(
        structure=_Sr2TiO4_structure(),
        frequencies=frequencies.view(np.float64).reshape(-1, 2),
        eigenvectors=_make_unitary_matrix(number_modes),
    )


def _piezoelectric_tensor():
    shape = (2, axes, axes, axes)
    data = np.arange(np.prod(shape)).reshape(shape)
    return raw.PiezoelectricTensor(electron=data[0], ion=data[1])


def _polarization():
    return raw.Polarization(electron=np.array((1, 2, 3)), ion=np.array((4, 5, 6)))


def _MD_energy(randomize: bool = False):
    labels = (
        "ion-electron   TOTEN",
        "kinetic energy EKIN",
        "kin. lattice   EKIN_LAT",
        "temperature    TEIN",
        "nose potential ES",
        "nose kinetic   EPS",
        "total energy   ETOTAL",
    )
    return _create_energy(labels, randomize=randomize)


def _relax_energy(randomize: bool = False):
    labels = (
        "free energy    TOTEN   ",
        "energy without entropy ",
        "energy(sigma->0)       ",
    )
    return _create_energy(labels, randomize=randomize)


def _afqmc_energy():
    labels = (
        "step            STEP    ",
        "One el. energy  E1      ",
        "Hartree energy  -DENC   ",
        "exchange        EXHF    ",
        "free energy     TOTEN   ",
        "free energy cap TOTENCAP",
        "weight          WEIGHT  ",
    )
    return _create_energy(labels)


def _create_energy(labels, randomize: bool = False):
    labels = np.array(labels, dtype="S")
    shape = (number_steps, len(labels))
    if randomize:
        return raw.Energy(labels=labels, values=np.random.random(shape))
    else:
        return raw.Energy(
            labels=labels, values=np.arange(np.prod(shape)).reshape(shape)
        )


def _qpoints():
    qpoints = _line_kpoints("line", "with_labels")
    qpoints.cell.lattice_vectors = qpoints.cell.lattice_vectors[-1]
    return qpoints


def _line_kpoints(mode, labels):
    line_length = 5
    GM = [0, 0, 0]
    Y = [0.5, 0.5, 0.0]
    A = [0, 0, 0.5]
    M = [0.5, 0.5, 0.5]
    coordinates = (
        np.linspace(GM, A, line_length),
        np.linspace(A, M, line_length),
        np.linspace(GM, Y, line_length),
        np.linspace(Y, M, line_length),
    )
    kpoints = raw.Kpoint(
        mode=mode,
        number=line_length,
        coordinates=np.concatenate(coordinates),
        weights=np.ones(len(coordinates)),
        cell=_Sr2TiO4_cell(),
    )
    if labels == "with_labels":
        kpoints.labels = _make_data([r"$\Gamma$", " M ", r"$\Gamma$", "Y", "M"])
        kpoints.label_indices = _make_data([1, 4, 5, 7, 8])
    return kpoints


def _slice_kpoints(mode):
    nkpx, nkpy, nkpz = (4, 3, 1)
    x = np.linspace(0, 1, nkpx, endpoint=False)
    y = np.linspace(0, 1, nkpy, endpoint=False)
    z = np.linspace(0, 1, nkpz, endpoint=False) + 1 / 8
    coordinates = np.array(list(itertools.product(x, y, z)))
    number_kpoints = len(coordinates) if mode[0] in ["e", b"e"[0]] else 0
    number_kpx = nkpx if mode[0] in ["e", b"e"[0]] else 0
    number_kpy = nkpy if mode[0] in ["e", b"e"[0]] else 0
    number_kpz = nkpz if mode[0] in ["e", b"e"[0]] else 0
    kpoints = raw.Kpoint(
        mode=mode,
        number=number_kpoints,
        number_x=number_kpx,
        number_y=number_kpy,
        number_z=number_kpz,
        coordinates=coordinates,
        weights=np.arange(len(coordinates)),
        cell=_Ba2PbO4_cell(),
    )
    return kpoints


def _grid_kpoints(mode, labels):
    nkpx, nkpy, nkpz = (4, 3, 4)
    x = np.linspace(0, 1, nkpx, endpoint=False)
    y = np.linspace(0, 1, nkpy, endpoint=False)
    z = np.linspace(0, 1, nkpz, endpoint=False) + 1 / 8
    coordinates = np.array(list(itertools.product(x, y, z)))
    number_kpoints = len(coordinates) if mode[0] in ["e", b"e"[0]] else 0
    kpoints = raw.Kpoint(
        mode=mode,
        number=number_kpoints,
        number_x=nkpx,
        number_y=nkpy,
        number_z=nkpz,
        coordinates=coordinates,
        weights=np.arange(len(coordinates)),
        cell=_Sr2TiO4_cell(),
    )
    if labels == "with_labels":
        kpoints.labels = _make_data(["foo", b"bar", "baz"])
        kpoints.label_indices = _make_data([9, 25, 40])
    return kpoints


def _local_moment(selection):
    lmax = 3 if selection != "noncollinear" else 4
    number_components = _number_components(selection)
    shape = (number_steps, number_components, number_atoms, lmax)
    moment = raw.LocalMoment(
        structure=_Fe3O4_structure(),
        spin_moments=_make_data(np.arange(np.prod(shape)).reshape(shape)),
    )
    if selection == "orbital_moments":
        remove_charge_and_s_component = moment.spin_moments[:, 1:, :, 1:]
        moment.orbital_moments = _make_data(np.sqrt(remove_charge_and_s_component))
    return moment


def _single_band(projectors):
    dispersion = _single_band_dispersion()
    return raw.Band(
        dispersion=dispersion,
        fermi_energy=0.0,
        occupations=np.array([np.linspace([1], [0], dispersion.eigenvalues.size)]),
        projectors=_Sr2TiO4_projectors(use_orbitals=False),
    )


def _single_band_dispersion():
    kpoints = _grid_kpoints("explicit", "no_labels")
    eigenvalues = np.array([np.linspace([0], [1], len(kpoints.coordinates))])
    return raw.Dispersion(kpoints, eigenvalues)


def _multiple_bands(projectors):
    dispersion = _multiple_bands_dispersion()
    shape = dispersion.eigenvalues.shape
    use_orbitals = projectors == "with_projectors"
    raw_band = raw.Band(
        dispersion=dispersion,
        fermi_energy=0.5,
        occupations=np.arange(np.prod(shape)).reshape(shape),
        projectors=_Sr2TiO4_projectors(use_orbitals),
    )
    if use_orbitals:
        number_orbitals = len(raw_band.projectors.orbital_types)
        shape = (single_spin, number_atoms, number_orbitals, *shape[1:])
        raw_band.projections = np.random.random(shape)
    return raw_band


def _multiple_bands_dispersion():
    kpoints = _grid_kpoints("explicit", "no_labels")
    shape = (single_spin, len(kpoints.coordinates), number_bands)
    eigenvalues = np.arange(np.prod(shape)).reshape(shape)
    return raw.Dispersion(kpoints, eigenvalues)


def _line_band(labels):
    dispersion = _line_dispersion(labels)
    shape = dispersion.eigenvalues.shape
    return raw.Band(
        dispersion=dispersion,
        fermi_energy=0.5,
        occupations=np.arange(np.prod(shape)).reshape(shape),
        projectors=_Sr2TiO4_projectors(use_orbitals=False),
    )


def _line_dispersion(labels):
    kpoints = _line_kpoints("line", labels)
    shape = (single_spin, len(kpoints.coordinates), number_bands)
    eigenvalues = np.arange(np.prod(shape)).reshape(shape)
    return raw.Dispersion(kpoints, eigenvalues)


def _spin_polarized_bands(projectors):
    dispersion = _spin_polarized_dispersion()
    shape = dispersion.eigenvalues.shape
    use_orbitals = projectors in ["with_projectors", "excess_orbitals"]
    raw_band = raw.Band(
        dispersion=dispersion,
        fermi_energy=0.0,
        occupations=np.arange(np.prod(shape)).reshape(shape),
        projectors=_Fe3O4_projectors(use_orbitals),
    )
    if use_orbitals:
        number_orbitals = len(raw_band.projectors.orbital_types)
        shape = (two_spins, number_atoms, number_orbitals, *shape[1:])
        raw_band.projections = np.random.random(shape)
    if projectors == "excess_orbitals":
        orbital_types = _make_orbital_types(use_orbitals, "s p d f g h i")
        raw_band.projectors.orbital_types = orbital_types
    return raw_band


def _spin_polarized_dispersion():
    kpoints = _grid_kpoints("explicit", "no_labels")
    kpoints.cell = _Fe3O4_cell()
    shape = (two_spins, len(kpoints.coordinates), number_bands)
    eigenvalues = np.arange(np.prod(shape)).reshape(shape)
    return raw.Dispersion(kpoints, eigenvalues)


def _noncollinear_bands(projectors):
    dispersion = _noncollinear_dispersion()
    shape = dispersion.eigenvalues.shape
    use_orbitals = projectors == "with_projectors"
    raw_band = raw.Band(
        dispersion=dispersion,
        fermi_energy=0.0,
        occupations=_make_arbitrary_data(shape),
        projectors=_Ba2PbO4_projectors(use_orbitals),
    )
    if use_orbitals:
        number_orbitals = len(raw_band.projectors.orbital_types)
        shape = (noncollinear, number_atoms, number_orbitals, *shape[1:])
        raw_band.projections = _make_arbitrary_data(shape)
    return raw_band


def _noncollinear_dispersion():
    kpoints = _line_kpoints("explicit", "no_labels")
    kpoints.cell = _Ba2PbO4_cell()
    shape = (noncollinear, len(kpoints.coordinates), number_bands)
    return raw.Dispersion(kpoints, eigenvalues=_make_arbitrary_data(shape))


def _spin_texture_bands(projectors):
    dispersion = _spin_texture_dispersion()
    shape = dispersion.eigenvalues.shape
    use_orbitals = projectors in ["with_projectors"]
    raw_band = raw.Band(
        dispersion=dispersion,
        fermi_energy=0.0,
        occupations=np.arange(np.prod(shape)).reshape(shape),
        projectors=_Ba2PbO4_projectors(use_orbitals),
    )
    if use_orbitals:
        number_orbitals = len(raw_band.projectors.orbital_types)
        shape = (4, number_atoms, number_orbitals, *shape[1:])
        raw_band.projections = np.random.random(shape)
    return raw_band


def _spin_texture_dispersion():
    kpoints = _slice_kpoints("explicit")
    kpoints.cell = _Ba2PbO4_cell()
    shape = (two_spins, len(kpoints.coordinates), number_bands)
    eigenvalues = np.arange(np.prod(shape)).reshape(shape)
    return raw.Dispersion(kpoints, eigenvalues)


def _workfunction(direction):
    shape = (number_points,)
    return raw.Workfunction(
        idipol=int(direction),
        distance=_make_arbitrary_data(shape),
        average_potential=_make_arbitrary_data(shape),
        vacuum_potential=_make_arbitrary_data(shape=(2,)),
        reference_potential=_bandgap("nonpolarized"),
        fermi_energy=1.234,
    )


def _Ba2PbO4_cell():
    lattice_vectors = [
        [4.34, 0.0, 0.0],
        [0.0, 4.34, 0.0],
        [-2.17, -2.17, 6.682],
    ]
    return raw.Cell(lattice_vectors=np.array(lattice_vectors), scale=1.0)


def _Ba2PbO4_dos(projectors):
    assert projectors == "noncollinear"
    energies = np.linspace(-4, 1, number_points)
    raw_dos = raw.Dos(
        fermi_energy=-1.3,
        energies=energies,
        dos=_make_arbitrary_data((noncollinear, number_points)),
        projectors=_Ba2PbO4_projectors(use_orbitals=True),
    )
    number_orbitals = len(raw_dos.projectors.orbital_types)
    shape = (noncollinear, number_atoms, number_orbitals, number_points)
    raw_dos.projections = _make_arbitrary_data(shape)
    return raw_dos


def _Ba2PbO4_projectors(use_orbitals):
    orbital_types = "s p d f"
    return raw.Projector(
        stoichiometry=_Ba2PbO4_stoichiometry(),
        orbital_types=_make_orbital_types(use_orbitals, orbital_types),
        number_spin_projections=4,
    )


def _Ba2PbO4_stoichiometry():
    return raw.Stoichiometry(
        number_ion_types=np.array((2, 1, 4)),
        ion_types=raw.VaspData(np.array(("Ba", "Pb", "O "), dtype="S")),
    )


def _electronic_minimization():
    random_convergence_data = np.random.rand(9, 3)
    iteration_number = np.arange(1, 10)[:, np.newaxis]
    ncg = np.random.randint(4, 10, (9, 1))
    random_rms = np.random.rand(9, 2)
    convergence_data = np.hstack(
        [iteration_number, random_convergence_data, ncg, random_rms]
    )
    convergence_data = raw.VaspData(convergence_data)
    label = raw.VaspData([b"N", b"E", b"dE", b"deps", b"ncg", b"rms", b"rms(c)"])
    is_elmin_converged = [0]
    return raw.ElectronicMinimization(
        convergence_data=convergence_data,
        label=label,
        is_elmin_converged=is_elmin_converged,
    )


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


def _Sr2TiO4_born_effective_charges():
    shape = (number_atoms, axes, axes)
    return raw.BornEffectiveCharge(
        structure=_Sr2TiO4_structure(),
        charge_tensors=np.arange(np.prod(shape)).reshape(shape),
    )


def _Sr2TiO4_cell():
    scale = raw.VaspData(6.9229)
    lattice_vectors = [
        [1.0, 0.0, 0.0],
        [0.678112209738693, 0.734958387251008, 0.0],
        [-0.839055341042049, -0.367478859090843, 0.401180037874301],
    ]
    return raw.Cell(
        lattice_vectors=np.array(number_steps * [lattice_vectors]), scale=scale
    )


def _Sr2TiO4_CONTCAR():
    structure = _Sr2TiO4_structure()
    structure.cell.lattice_vectors = structure.cell.lattice_vectors[-1]
    structure.positions = structure.positions[-1]
    return raw.CONTCAR(structure=structure, system=b"Sr2TiO4")


def _Sr2TiO4_density():
    structure = _Sr2TiO4_structure()
    grid = (1, *grid_dimensions)
    return raw.Density(structure=structure, charge=_make_arbitrary_data(grid))


def _Sr2TiO4_dos(projectors):
    energies = np.linspace(-1, 3, number_points)
    use_orbitals = projectors == "with_projectors"
    raw_dos = raw.Dos(
        fermi_energy=1.372,
        energies=energies,
        dos=np.array([energies**2]),
        projectors=_Sr2TiO4_projectors(use_orbitals),
    )
    if use_orbitals:
        number_orbitals = len(raw_dos.projectors.orbital_types)
        shape = (single_spin, number_atoms, number_orbitals, number_points)
        raw_dos.projections = np.random.random(shape)
    return raw_dos


def _Sr2TiO4_exciton_density():
    shape = (number_excitons, *grid_dimensions)
    exciton_charge = _make_arbitrary_data(shape)
    return raw.ExcitonDensity(
        structure=_Sr2TiO4_structure(), exciton_charge=exciton_charge
    )


def _Sr2TiO4_exciton_eigenvector():
    dispersion = _multiple_bands_dispersion()
    number_kpoints = len(dispersion.kpoints.coordinates)
    shape = (single_spin, number_kpoints, number_conduction_bands, number_valence_bands)
    bse_index = np.arange(np.prod(shape)).reshape(shape)
    number_transitions = bse_index.size
    shape = (number_eigenvectors, number_transitions, complex_)
    eigenvectors = np.random.uniform(0, 20, shape)
    return raw.ExcitonEigenvector(
        dispersion=dispersion,
        fermi_energy=0.2,
        bse_index=raw.VaspData(bse_index),
        eigenvectors=raw.VaspData(eigenvectors),
        first_valence_band=raw.VaspData(np.array([1])),
        first_conduction_band=raw.VaspData(np.array([3])),
    )


def _Sr2TiO4_force_constants(use_selective_dynamics):
    shape = (axes * number_atoms, axes * number_atoms)
    force_constants = _make_arbitrary_data(shape, seed=51609352)
    if use_selective_dynamics:
        mask = 3 * [True] + 5 * [False] + 5 * [True] + 6 * [False] + 2 * [True]
        force_constants = force_constants[mask][:, mask]
        selective_dynamics = np.reshape(mask, (number_atoms, axes))
    else:
        selective_dynamics = _make_arbitrary_data(None, present=False)
    return raw.ForceConstant(
        structure=_Sr2TiO4_structure(),
        force_constants=0.5 * (force_constants + force_constants[:].T),
        selective_dynamics=selective_dynamics,
    )


def _Sr2TiO4_forces(randomize):
    shape = (number_steps, number_atoms, axes)
    if randomize:
        forces = np.random.random(shape)
    else:
        forces = np.arange(np.prod(shape)).reshape(shape)
    return raw.Force(
        structure=_Sr2TiO4_structure(),
        forces=forces,
    )


def _Sr2TiO4_internal_strain():
    shape = (number_atoms, axes, axes, axes)
    return raw.InternalStrain(
        structure=_Sr2TiO4_structure(),
        internal_strain=np.arange(np.prod(shape)).reshape(shape),
    )


def _Sr2TiO4_nics():
    structure = _Sr2TiO4_structure()
    grid = (9, *grid_dimensions)
    return raw.Nics(structure=structure, nics_grid=_make_arbitrary_data(grid))


def _Sr2TiO4_pair_correlation():
    labels = ("total", "Sr~Sr", "Sr~Ti", "Sr~O", "Ti~Ti", "Ti~O", "O~O")
    shape = (number_steps, len(labels), number_points)
    data = np.arange(np.prod(shape)).reshape(shape)
    return raw.PairCorrelation(
        distances=np.arange(number_points), function=data, labels=labels
    )


def _Sr2TiO4_potential(included_potential):
    structure = _Sr2TiO4_structure()
    shape = (1, *grid_dimensions)
    include_xc = included_potential in ("xc", "all")
    include_hartree = included_potential in ("hartree", "all")
    include_ionic = included_potential in ("ionic", "all")
    return raw.Potential(
        structure=structure,
        total_potential=_make_arbitrary_data(shape),
        xc_potential=_make_arbitrary_data(shape, present=include_xc),
        hartree_potential=_make_arbitrary_data(shape, present=include_hartree),
        ionic_potential=_make_arbitrary_data(shape, present=include_ionic),
    )


def _Sr2TiO4_projectors(use_orbitals):
    orbital_types = "s py pz px dxy dyz dz2 dxz x2-y2 fy3x2 fxyz fyz2 fz3 fxz2 fzx2 fx3"
    return raw.Projector(
        stoichiometry=_Sr2TiO4_stoichiometry(),
        orbital_types=_make_orbital_types(use_orbitals, orbital_types),
        number_spin_projections=1,
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
    repetitions = (number_steps, 1, 1)
    positions = [
        [0.64529, 0.64529, 0.0],
        [0.35471, 0.35471, 0.0],
        [0.00000, 0.00000, 0.0],
        [0.84178, 0.84178, 0.0],
        [0.15823, 0.15823, 0.0],
        [0.50000, 0.00000, 0.5],
        [0.00000, 0.50000, 0.5],
    ]
    return raw.Structure(
        stoichiometry=_Sr2TiO4_stoichiometry(has_ion_types),
        cell=_Sr2TiO4_cell(),
        positions=np.tile(positions, repetitions),
    )


def _Sr2TiO4_stoichiometry(has_ion_types=True):
    if has_ion_types:
        return raw.Stoichiometry(
            number_ion_types=np.array((2, 1, 4)),
            ion_types=raw.VaspData(np.array(("Sr", "Ti", "O "), dtype="S")),
        )
    else:
        return raw.Stoichiometry(
            number_ion_types=raw.VaspData(np.array((2, 1, 4))),
            ion_types=raw.VaspData(None),
        )


def _Sr2TiO4_velocity():
    shape = (number_steps, number_atoms, axes)
    velocities = np.arange(np.prod(shape)).reshape(shape)
    return raw.Velocity(structure=_Sr2TiO4_structure(), velocities=velocities)


def _Fe3O4_cell():
    lattice_vectors = [
        [5.1427, 0.0, 0.0],
        [0.0, 3.0588, 0.0],
        [-1.3633791448, 0.0, 5.0446102592],
    ]
    scaling = np.linspace(0.98, 1.01, number_steps)
    lattice_vectors = np.multiply.outer(scaling, lattice_vectors)
    return raw.Cell(lattice_vectors, scale=raw.VaspData(None))


def _Fe3O4_CONTCAR():
    structure = _Fe3O4_structure()
    structure.cell.lattice_vectors = structure.cell.lattice_vectors[-1]
    structure.positions = structure.positions[-1]
    even_numbers = np.arange(structure.positions.size) % 2 == 0
    selective_dynamics = even_numbers.reshape(structure.positions.shape)
    lattice_velocities = 0.1 * structure.cell.lattice_vectors**2 - 0.3
    shape = structure.positions.shape
    ion_velocities = np.sqrt(np.arange(np.prod(shape)).reshape(shape))
    return raw.CONTCAR(
        structure=structure,
        system="Fe3O4",
        selective_dynamics=raw.VaspData(selective_dynamics),
        lattice_velocities=raw.VaspData(lattice_velocities),
        ion_velocities=raw.VaspData(ion_velocities),
    )


def _Fe3O4_density(selection):
    structure = _Fe3O4_structure()
    grid = (_number_components(selection), *grid_dimensions)
    return raw.Density(structure=structure, charge=_make_arbitrary_data(grid))


def _Fe3O4_dos(projectors):
    energies = np.linspace(-2, 2, number_points)
    use_orbitals = projectors in ["with_projectors", "excess_orbitals"]
    raw_dos = raw.Dos(
        fermi_energy=-0.137,
        energies=energies,
        dos=np.array(((energies + 0.5) ** 2, (energies - 0.5) ** 2)),
        projectors=_Fe3O4_projectors(use_orbitals),
    )
    if use_orbitals:
        number_orbitals = len(raw_dos.projectors.orbital_types)
        shape = (two_spins, number_atoms, number_orbitals, number_points)
        raw_dos.projections = np.random.random(shape)
    if projectors == "excess_orbitals":
        orbital_types = _make_orbital_types(use_orbitals, "s p d f g h i")
        raw_dos.projectors.orbital_types = orbital_types
    return raw_dos


def _Fe3O4_forces(randomize):
    shape = (number_steps, number_atoms, axes)
    if randomize:
        forces = np.random.random(shape)
    else:
        forces = np.arange(np.prod(shape)).reshape(shape)
    return raw.Force(structure=_Fe3O4_structure(), forces=forces)


def _Fe3O4_nics():
    structure = _Fe3O4_structure()
    seed_nics = 4782838
    seed_pos = 6375861
    positions_shape = (axes, number_points)
    nics_shape = (number_points, axes, axes)
    nics_data = np.array(_make_arbitrary_data(nics_shape, seed=seed_nics))
    # intentionally make values very small to check their output
    nics_data[4, 1, 0] = 1e-108
    nics_data[9, 0, 2] = -1e-15  # should be rounded
    nics_data[11, 2, 1] = 1e-14  # should still be there
    return raw.Nics(
        structure=structure,
        nics_points=raw.VaspData(nics_data),
        positions=_make_arbitrary_data(positions_shape, seed=seed_pos),
    )


def _Fe3O4_potential(selection, included_potential):
    structure = _Fe3O4_structure()
    shape_polarized = (_number_components(selection), *grid_dimensions)
    shape_trivial = (1, *grid_dimensions)
    include_xc = included_potential in ("xc", "all")
    include_hartree = included_potential in ("hartree", "all")
    include_ionic = included_potential in ("ionic", "all")
    return raw.Potential(
        structure=structure,
        total_potential=_make_arbitrary_data(shape_polarized),
        xc_potential=_make_arbitrary_data(shape_polarized, present=include_xc),
        hartree_potential=_make_arbitrary_data(shape_trivial, present=include_hartree),
        ionic_potential=_make_arbitrary_data(shape_trivial, present=include_ionic),
    )


def _Fe3O4_projectors(use_orbitals):
    return raw.Projector(
        stoichiometry=_Fe3O4_stoichiometry(),
        orbital_types=_make_orbital_types(use_orbitals, "s p d f"),
        number_spin_projections=2,
    )


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
    positions = [
        [0.00000, 0.0, 0.00000],
        [0.50000, 0.0, 0.50000],
        [0.00000, 0.5, 0.50000],
        [0.78745, 0.0, 0.28152],
        [0.26310, 0.5, 0.27611],
        [0.21255, 0.0, 0.71848],
        [0.73690, 0.5, 0.72389],
    ]
    shift = np.linspace(-0.02, 0.01, number_steps)
    return raw.Structure(
        stoichiometry=_Fe3O4_stoichiometry(),
        cell=_Fe3O4_cell(),
        positions=np.add.outer(shift, positions),
    )


def _Fe3O4_stoichiometry():
    return raw.Stoichiometry(
        number_ion_types=np.array((3, 4)), ion_types=np.array(("Fe", "O "), dtype="S")
    )


def _Fe3O4_velocity():
    shape = (number_steps, number_atoms, axes)
    velocities = np.arange(np.prod(shape)).reshape(shape)
    return raw.Velocity(structure=_Fe3O4_structure(), velocities=velocities)


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
    return raw.Stoichiometry(
        number_ion_types=np.array((2, 1, 1, 1, 2)),
        ion_types=np.array(("Ca", "As", "Br", "Ca", "Br"), dtype="S"),
    )


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


def _current_density(selection):
    if selection == "all":
        valid_indices = ("x", "y", "z")
    else:
        valid_indices = [selection]
    shape = (axes, *grid_dimensions)
    current_density = [_make_arbitrary_data(shape) for _ in valid_indices]
    return raw.CurrentDensity(
        valid_indices=valid_indices,
        structure=_Fe3O4_structure(),
        current_density=current_density,
    )


def _electron_phonon_chemical_potential(selection="carrier_den"):
    number_temps = 6
    return raw.ElectronPhononChemicalPotential(
        fermi_energy=0,
        carrier_density=_make_arbitrary_data([number_samples]),
        temperatures=_make_arbitrary_data([number_temps]),
        chemical_potential=_make_arbitrary_data([number_samples, number_temps]),
        carrier_per_cell=_make_arbitrary_data(
            [number_samples], selection == "carrier_per_cell"
        ),
        mu=_make_arbitrary_data([number_samples], selection == "mu"),
        carrier_den=_make_arbitrary_data([number_samples], selection == "carrier_den"),
    )


def _electron_phonon_self_energy(selection):
    nspin = 1
    nkpoint = 2
    nband = 4
    band_start = 1
    # mock band_kpoint_spin_index array
    band_kpoint_spin_index_shape = [nspin, nkpoint, nband]
    band_kpoint_spin_index = np.full(band_kpoint_spin_index_shape, -1)
    ibks = 0
    for isp in range(nspin):
        for ikpt in range(nkpoint):
            for iband in range(nband):
                ibks = ibks + 1
                band_kpoint_spin_index[isp, ikpt, iband] = ibks
    # mock fan and dw
    nbks = np.count_nonzero(band_kpoint_spin_index != -1)
    nw = 1  # number of frequencies at which the fan self-energy is evaluated
    ntemps = 6
    nbands_sum = 12
    scattering_approximation = "MRTA_TAU"
    fan_shape = [nbks, nw, ntemps]
    debye_waller_shape = [nbks, ntemps]
    return raw.ElectronPhononSelfEnergy(
        valid_indices=range(number_samples),
        id_name=["selfen_delta", "nbands_sum", "selfen_muij", "selfen_approx"],
        id_size=[1, number_samples, 1, 1],
        nbands_sum=np.array([nbands_sum for _ in range(number_samples)]),
        delta=np.array([0 for _ in range(number_samples)]),
        scattering_approximation=[
            scattering_approximation for _ in range(number_samples)
        ],
        chemical_potential=_electron_phonon_chemical_potential(selection),
        id_index=[[1, sample + 1, 1] for sample in range(number_samples)],
        eigenvalues=_make_arbitrary_data(band_kpoint_spin_index_shape),
        debye_waller=[
            _make_arbitrary_data(debye_waller_shape) for _ in range(number_samples)
        ],
        fan=[_make_arbitrary_data(fan_shape) for _ in range(number_samples)],
        band_kpoint_spin_index=np.array(
            [band_kpoint_spin_index for _ in range(number_samples)]
        ),
        band_start=np.array([band_start for _ in range(number_samples)]),
    )


def _electron_phonon_band_gap(selection):
    number_components = 3 if selection == "collinear" else 1
    number_temps = 6
    shape_gap = [number_samples, number_components]
    shape_renorm = [number_samples, number_components, number_temps]
    shape_temperature = [number_samples, number_temps]
    scattering_approximation = "SERTA"
    return raw.ElectronPhononBandgap(
        valid_indices=range(number_samples),
        nbands_sum=_make_data(np.linspace(10, 100, number_samples, dtype=np.int32)),
        delta=_make_arbitrary_data([number_samples]),
        chemical_potential=_electron_phonon_chemical_potential(),
        scattering_approximation=[
            scattering_approximation for _ in range(number_samples)
        ],
        fundamental_renorm=_make_arbitrary_data(shape_renorm),
        direct_renorm=_make_arbitrary_data(shape_renorm),
        fundamental=_make_arbitrary_data(shape_gap),
        direct=_make_arbitrary_data(shape_gap),
        temperatures=_make_arbitrary_data(shape_temperature),
    )


def _electron_phonon_transport(selection):
    # mock transport_function
    nw = 1  # number of frequencies at which the fan self-energy is evaluated
    ntemps = 6
    nbands_sum = 12
    scattering_approximation = "MRTA_TAU"
    return raw.ElectronPhononTransport(
        valid_indices=range(number_samples),
        id_name=["selfen_delta", "nbands_sum", "selfen_muij", "selfen_approx"],
        id_size=[1, number_samples, 1],
        nbands_sum=np.array([nbands_sum for _ in range(number_samples)]),
        self_energy=_electron_phonon_self_energy(selection),
        chemical_potential=_electron_phonon_chemical_potential(selection),
        id_index=[[1, sample + 1, 1] for sample in range(number_samples)],
        delta=np.array([0 for _ in range(number_samples)]),
        temperatures=[np.linspace(0, 500, 6) for _ in range(number_samples)],
        transport_function=_make_arbitrary_data([number_samples, ntemps, nw, 3, 3]),
        mobility=_make_arbitrary_data([number_samples, ntemps, 3, 3]),
        seebeck=_make_arbitrary_data([number_samples, ntemps, 3, 3]),
        peltier=_make_arbitrary_data([number_samples, ntemps, 3, 3]),
        electronic_conductivity=_make_arbitrary_data([number_samples, ntemps, 3, 3]),
        electronic_thermal_conductivity=_make_arbitrary_data(
            [number_samples, ntemps, 3, 3]
        ),
        scattering_approximation=[
            scattering_approximation for _ in range(number_samples)
        ],
    )


def _make_unitary_matrix(n, seed=None):
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n, n))
    unitary_matrix, _ = np.linalg.qr(matrix)
    return raw.VaspData(unitary_matrix)


def _make_arbitrary_data(shape, present=True, seed=None):
    if present:
        rng = np.random.default_rng(seed)
        data = 10 * rng.standard_normal(shape)
        return raw.VaspData(data)
    else:
        return raw.VaspData(None)


def _make_data(data):
    return raw.VaspData(np.array(data))


def _make_orbital_types(use_orbitals, orbital_types):
    if use_orbitals:
        return raw.VaspData(np.array(orbital_types.split(), dtype="S"))
    else:
        return raw.VaspData(None)
