# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal_nulp

from py4vasp import exception, raw

number_steps = 4
number_atoms = 7
number_points = 50
number_bands = 3
number_valence_bands = 2
number_conduction_bands = 1
number_eigenvectors = 5
single_spin = 1
two_spins = 2
axes = 3
complex_ = 2
number_modes = axes * number_atoms


class _Assert:
    @staticmethod
    def allclose(actual, desired):
        if _is_none(actual):
            assert _is_none(desired)
        else:
            actual, desired = np.broadcast_arrays(actual, desired)
            actual, mask_actual = _finite_subset(actual)
            desired, mask_desired = _finite_subset(desired)
            assert np.all(mask_actual == mask_desired)
            assert_array_almost_equal_nulp(actual, desired, 10)


def _is_none(data):
    if isinstance(data, raw.VaspData):
        return data.is_none()
    else:
        return data is None


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
        else:
            raise exception.NotImplemented()

    @staticmethod
    def born_effective_charge(selection):
        if selection == "Sr2TiO4":
            return _Sr2TiO4_born_effective_charges()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def density(selection):
        return _Fe3O4_density(selection)

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
        else:
            raise exception.NotImplemented()

    @staticmethod
    def elastic_modulus(selection):
        return _elastic_modulus()

    @staticmethod
    def energy(selection):
        return _energy()

    @staticmethod
    def fatband(selection):
        return _Sr2TiO4_fatband()

    @staticmethod
    def force_constant(selection):
        if selection == "Sr2TiO4":
            return _Sr2TiO4_force_constants()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def force(selection):
        if selection == "Sr2TiO4":
            return _Sr2TiO4_forces()
        elif selection == "Fe3O4":
            return _Fe3O4_forces()
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
    def magnetism(selection):
        return _magnetism(_number_components(selection))

    @staticmethod
    def pair_correlation(selection):
        return _Sr2TiO4_pair_correlation()

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
    def projector(selection):
        if selection == "Sr2TiO4":
            return _Sr2TiO4_projectors(use_orbitals=True)
        elif selection == "Fe3O4":
            return _Fe3O4_projectors(use_orbitals=True)
        elif selection == "without_orbitals":
            return _Sr2TiO4_projectors(use_orbitals=False)
        else:
            raise exception.NotImplemented()

    @staticmethod
    def stress(selection):
        if selection == "Sr2TiO4":
            return _Sr2TiO4_stress()
        elif selection == "Fe3O4":
            return _Fe3O4_stress()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def structure(selection):
        if selection == "Sr2TiO4":
            return _Sr2TiO4_structure()
        elif selection == "Fe3O4":
            return _Fe3O4_structure()
        else:
            raise exception.NotImplemented()

    @staticmethod
    def topology(selection):
        if selection == "Sr2TiO4":
            return _Sr2TiO4_topology()
        elif selection == "Fe3O4":
            return _Fe3O4_topology()
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


@pytest.fixture
def raw_data():
    return RawDataFactory


def _number_components(selection):
    if selection == "collinear":
        return 2
    elif selection == "noncollinear":
        return 4
    elif selection == "charge_only":
        return 1
    else:
        raise exception.NotImplemented()


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


def _Sr2TiO4_pair_correlation():
    labels = ("total", "Sr~Sr", "Sr~Ti", "Sr~O", "Ti~Ti", "Ti~O", "O~O")
    shape = (number_steps, len(labels), number_points)
    data = np.arange(np.prod(shape)).reshape(shape)
    return raw.PairCorrelation(
        distances=np.arange(number_points), function=data, labels=labels
    )


def _phonon_band():
    dispersion = _phonon_dispersion()
    shape = (*dispersion.eigenvalues.shape, number_atoms, axes, complex_)
    return raw.PhononBand(
        dispersion=dispersion,
        topology=_Sr2TiO4_topology(),
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
    return raw.PhononDos(energies, dos, projections, _Sr2TiO4_topology())


def _piezoelectric_tensor():
    shape = (2, axes, axes, axes)
    data = np.arange(np.prod(shape)).reshape(shape)
    return raw.PiezoelectricTensor(electron=data[0], ion=data[1])


def _polarization():
    return raw.Polarization(electron=np.array((1, 2, 3)), ion=np.array((4, 5, 6)))


def _energy():
    labels = ("ion-electron   TOTEN    ", "kinetic energy EKIN", "temperature    TEIN")
    labels = np.array(labels, dtype="S")
    shape = (number_steps, len(labels))
    return raw.Energy(labels=labels, values=np.arange(np.prod(shape)).reshape(shape))


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


def _grid_kpoints(mode, labels):
    x = np.linspace(0, 1, 4, endpoint=False)
    y = np.linspace(0, 1, 3, endpoint=False)
    z = np.linspace(0, 1, 4, endpoint=False) + 1 / 8
    coordinates = np.array(list(itertools.product(x, y, z)))
    number_kpoints = len(coordinates) if mode[0] in ["e", b"e"[0]] else 0
    kpoints = raw.Kpoint(
        mode=mode,
        number=number_kpoints,
        coordinates=coordinates,
        weights=np.arange(len(coordinates)),
        cell=_Sr2TiO4_cell(),
    )
    if labels == "with_labels":
        kpoints.labels = _make_data(["foo", b"bar", "baz"])
        kpoints.label_indices = _make_data([9, 25, 40])
    return kpoints


def _magnetism(number_components):
    lmax = 3
    shape = (number_steps, number_components, number_atoms, lmax)
    return raw.Magnetism(
        structure=_Fe3O4_structure(), moments=np.arange(np.prod(shape)).reshape(shape)
    )


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


def _Sr2TiO4_born_effective_charges():
    shape = (number_atoms, axes, axes)
    return raw.BornEffectiveCharge(
        structure=_Sr2TiO4_structure(),
        charge_tensors=np.arange(np.prod(shape)).reshape(shape),
    )


def _Sr2TiO4_cell():
    scale = 6.9229
    lattice_vectors = [
        [1.0, 0.0, 0.0],
        [0.678112209738693, 0.734958387251008, 0.0],
        [-0.839055341042049, -0.367478859090843, 0.401180037874301],
    ]
    return raw.Cell(
        lattice_vectors=scale * np.array(number_steps * [lattice_vectors]), scale=scale
    )


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


def _Sr2TiO4_fatband():
    dispersion = _multiple_bands_dispersion()
    number_kpoints = len(dispersion.kpoints.coordinates)
    shape = (single_spin, number_kpoints, number_conduction_bands, number_valence_bands)
    bse_index = np.arange(np.prod(shape)).reshape(shape)
    number_transitions = bse_index.size
    shape = (number_eigenvectors, number_transitions, complex_)
    fatbands = np.random.uniform(0, 20, shape)
    return raw.Fatband(
        dispersion=dispersion,
        fermi_energy=0.2,
        bse_index=raw.VaspData(bse_index),
        fatbands=raw.VaspData(fatbands),
        first_valence_band=raw.VaspData(np.array([1])),
        first_conduction_band=raw.VaspData(np.array([3])),
    )


def _Sr2TiO4_force_constants():
    shape = (axes * number_atoms, axes * number_atoms)
    return raw.ForceConstant(
        structure=_Sr2TiO4_structure(),
        force_constants=np.arange(np.prod(shape)).reshape(shape),
    )


def _Sr2TiO4_forces():
    shape = (number_steps, number_atoms, axes)
    return raw.Force(
        structure=_Sr2TiO4_structure(), forces=np.arange(np.prod(shape)).reshape(shape)
    )


def _Sr2TiO4_internal_strain():
    shape = (number_atoms, axes, axes, axes)
    return raw.InternalStrain(
        structure=_Sr2TiO4_structure(),
        internal_strain=np.arange(np.prod(shape)).reshape(shape),
    )


def _Sr2TiO4_projectors(use_orbitals):
    orbital_types = "s py pz px dxy dyz dz2 dxz x2-y2 fy3x2 fxyz fyz2 fz3 fxz2 fzx2 fx3"
    return raw.Projector(
        topology=_Sr2TiO4_topology(),
        orbital_types=_make_orbital_types(use_orbitals, orbital_types),
        number_spins=1,
    )


def _Sr2TiO4_stress():
    shape = (number_steps, axes, axes)
    return raw.Stress(
        structure=_Sr2TiO4_structure(), stress=np.arange(np.prod(shape)).reshape(shape)
    )


def _Sr2TiO4_structure():
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
        topology=_Sr2TiO4_topology(),
        cell=_Sr2TiO4_cell(),
        positions=np.tile(positions, repetitions),
    )


def _Sr2TiO4_topology():
    return raw.Topology(
        number_ion_types=np.array((2, 1, 4)),
        ion_types=np.array(("Sr", "Ti", "O "), dtype="S"),
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
    return raw.Cell(lattice_vectors=np.multiply.outer(scaling, lattice_vectors))


def _Fe3O4_density(selection):
    parts = selection.split()
    structure = RawDataFactory.structure(parts[0])
    grid = (_number_components(parts[1]), 10, 12, 14)
    return raw.Density(
        structure=structure, charge=np.arange(np.prod(grid)).reshape(grid)
    )


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


def _Fe3O4_forces():
    shape = (number_steps, number_atoms, axes)
    return raw.Force(
        structure=_Fe3O4_structure(), forces=np.arange(np.prod(shape)).reshape(shape)
    )


def _Fe3O4_projectors(use_orbitals):
    return raw.Projector(
        topology=_Fe3O4_topology(),
        orbital_types=_make_orbital_types(use_orbitals, "s p d f"),
        number_spins=2,
    )


def _Fe3O4_stress():
    shape = (number_steps, axes, axes)
    return raw.Stress(
        structure=_Fe3O4_structure(), stress=np.arange(np.prod(shape)).reshape(shape)
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
        topology=_Fe3O4_topology(),
        cell=_Fe3O4_cell(),
        positions=np.add.outer(shift, positions),
    )


def _Fe3O4_topology():
    return raw.Topology(
        number_ion_types=np.array((3, 4)), ion_types=np.array(("Fe", "O "), dtype="S")
    )


def _Fe3O4_velocity():
    shape = (number_steps, number_atoms, axes)
    velocities = np.arange(np.prod(shape)).reshape(shape)
    return raw.Velocity(structure=_Fe3O4_structure(), velocities=velocities)


def _make_data(data):
    return raw.VaspData(np.array(data))


def _make_orbital_types(use_orbitals, orbital_types):
    if use_orbitals:
        return raw.VaspData(np.array(orbital_types.split(), dtype="S"))
    else:
        return raw.VaspData(None)
