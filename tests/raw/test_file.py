from py4vasp.raw import *
import py4vasp.exceptions as exception
import contextlib
import pytest
import h5py
import os
import pathlib
import numpy as np
import itertools
import inspect
from collections import namedtuple
from numbers import Number, Integral
from unittest.mock import patch

num_spins = 2
num_energies = 20
num_kpoints = 10
num_bands = 3
num_atoms = 10  # sum(range(5))
num_steps = 15
num_components = 2  # charge and magnetization
complex = 2
axes = 3
lmax = 3
fermi_energy = 0.123
default_options = ((True,), (False,))
default_sources = ("default",)
reference_version = RawVersion(major=99, minor=98, patch=97)

SetupTest = namedtuple(
    "SetupTest",
    "directory, options, sources, create_reference, write_reference, check_actual",
)


@contextlib.contextmanager
def working_directory(path):
    """A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit."""
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def test_file_as_context(tmp_path):
    filename = tmp_path / "test.h5"
    h5f = h5py.File(filename, "w")
    h5f.close()
    with File(filename) as file:
        assert not file.closed
        h5f = file._h5f
    # check that file is closed and accessing it raises ValueError
    assert file.closed
    with pytest.raises(ValueError):
        h5f.file
    with pytest.raises(exception.FileAccessError):
        file.version


def test_nonexisting_file():
    with pytest.raises(exception.FileAccessError):
        File()


def test_file_from_path(tmp_path):
    with patch("h5py.File") as mock_h5:
        with File(tmp_path) as file:
            assert not file.closed
        assert file.closed
        mock_h5.assert_called_once_with(tmp_path / File.default_filename, "r")
        mock_h5.reset_mock()
        File(str(tmp_path))
        mock_h5.assert_called_once_with(tmp_path / File.default_filename, "r")


def test_file_path(tmp_path):
    with patch("h5py.File") as mock_h5, working_directory(tmp_path):
        pathlib.Path(File.default_filename).touch()
        assert File().path == tmp_path
        directory = tmp_path / "subdirectory"
        pathlib.Path(directory).mkdir()
        pathlib.Path(directory / File.default_filename).touch()
        assert File(directory).path == directory
        filename = directory / "custom.h5"
        pathlib.Path(filename).touch()
        assert File(filename).path == directory


def generic_test(setup):
    with working_directory(setup.directory):
        for options in setup.options:
            for source in setup.sources:
                run_test(setup, options, source)


def run_test(setup, options, source):
    use_default, *additional_tags = options
    reference = setup.create_reference(*additional_tags)
    h5f, filename = open_h5_file(use_default)
    write_version(h5f, reference_version)
    setup.write_reference(h5f, reference, source)
    h5f.close()
    file = open_vasp_file(use_default, filename)
    setup.check_actual(file, reference, source)
    file.close()  # must be after comparison, because file is read lazily


def open_h5_file(use_default):
    filename = "vaspout.h5" if use_default else "generic_filename.h5"
    return h5py.File(filename, "w"), filename


def open_vasp_file(use_default, filename):
    return File() if use_default else File(filename)


def test_version(tmpdir):
    do_nothing = lambda *_: None
    setup = SetupTest(
        directory=tmpdir,
        options=default_options,
        sources=default_sources,
        create_reference=do_nothing,
        write_reference=do_nothing,
        check_actual=check_version,
    )
    generic_test(setup)


def write_version(h5f, version):
    h5f["version/major"] = version.major
    h5f["version/minor"] = version.minor
    h5f["version/patch"] = version.patch


def check_version(object, *_):
    assert object.version == reference_version


def get_actual_and_check_version(file, attribute, source):
    data_dict = getattr(file, attribute)
    check_version(data_dict)
    return data_dict[source]


def test_dos(tmpdir):
    setup = SetupTest(
        directory=tmpdir,
        options=itertools.product((True, False), repeat=3),
        sources=default_sources + ("kpoints_opt",),
        create_reference=reference_dos,
        write_reference=write_dos,
        check_actual=check_dos,
    )
    generic_test(setup)


def reference_dos(use_dos, use_projectors):
    shape = (num_spins, num_energies)
    return RawDos(
        fermi_energy=fermi_energy,
        energies=np.arange(num_energies) if use_dos else None,
        dos=np.arange(np.prod(shape)).reshape(shape),
        projectors=reference_projectors() if use_projectors else None,
    )


def write_dos(h5f, dos, source):
    suffix = source_suffix(source)
    if dos.energies is None:
        return
    h5f[f"results/electron_dos{suffix}/efermi"] = dos.fermi_energy
    h5f[f"results/electron_dos{suffix}/energies"] = dos.energies
    h5f[f"results/electron_dos{suffix}/dos"] = dos.dos
    if dos.projectors is not None:
        write_projectors(h5f, dos.projectors, source)
    if dos.projections is not None:
        h5f[f"results/electron_dos{suffix}/dospar"] = proj.dos


def check_dos(file, reference, source):
    actual = get_actual_and_check_version(file, "dos", source)
    if reference.energies is not None:
        assert actual == reference
        assert isinstance(actual.fermi_energy, Number)
    else:
        assert actual is None


def test_band(tmpdir):
    setup = SetupTest(
        directory=tmpdir,
        options=itertools.product((True, False), repeat=3),
        sources=default_sources + ("kpoints_opt",),
        create_reference=reference_band,
        write_reference=write_band,
        check_actual=check_band,
    )
    generic_test(setup)


def reference_band(use_projectors, use_labels):
    shape_eval = (num_spins, num_kpoints, num_bands)
    shape_proj = (num_spins, num_atoms, lmax, num_kpoints, num_bands)
    band = RawBand(
        fermi_energy=fermi_energy,
        kpoints=reference_kpoints(use_labels),
        eigenvalues=np.arange(np.prod(shape_eval)).reshape(shape_eval),
        occupations=np.arange(np.prod(shape_eval)).reshape(shape_eval),
    )
    if use_projectors:
        band.projectors = reference_projectors()
        band.projections = np.arange(np.prod(shape_proj)).reshape(shape_proj)
    return band


def write_band(h5f, band, source):
    suffix = source_suffix(source)
    h5f[f"results/electron_dos{suffix}/efermi"] = band.fermi_energy
    h5f[f"results/electron_eigenvalues{suffix}/eigenvalues"] = band.eigenvalues
    h5f[f"results/electron_eigenvalues{suffix}/fermiweights"] = band.occupations
    write_kpoints(h5f, band.kpoints, source)
    if band.projectors is not None:
        write_projectors(h5f, band.projectors, source)
    if band.projections is not None:
        h5f[f"results/projectors{suffix}/par"] = band.projections


def check_band(file, reference, source):
    actual = get_actual_and_check_version(file, "band", source)
    assert actual == reference
    assert isinstance(actual.fermi_energy, Number)


def test_projectors(tmpdir):
    setup = SetupTest(
        directory=tmpdir,
        options=default_options,
        sources=default_sources + ("kpoints_opt",),
        create_reference=reference_projectors,
        write_reference=write_projectors,
        check_actual=check_projectors,
    )
    generic_test(setup)


def reference_projectors():
    shape_dos = (num_spins, num_atoms, lmax, num_energies)
    return RawProjectors(
        topology=reference_topology(),
        orbital_types=np.array(["s", "p", "d", "f"], dtype="S"),
        number_spins=num_spins,
    )


def write_projectors(h5f, proj, source):
    suffix = source_suffix(source)
    write_topology(h5f, proj.topology, source)
    h5f[f"results/projectors{suffix}/lchar"] = proj.orbital_types
    key = f"results/electron_eigenvalues{suffix}/eigenvalues"
    if key not in h5f:
        h5f[key] = np.arange(proj.number_spins)


def check_projectors(file, reference, source):
    actual = get_actual_and_check_version(file, "projectors", source)
    assert actual == reference
    assert isinstance(actual.number_spins, Integral)


def test_topoplogy(tmpdir):
    setup = SetupTest(
        directory=tmpdir,
        options=default_options,
        sources=default_sources,
        create_reference=reference_topology,
        write_reference=write_topology,
        check_actual=check_topology,
    )
    generic_test(setup)


def reference_topology():
    return RawTopology(
        number_ion_types=np.arange(5),
        ion_types=np.array(["B", "C", "N", "O", "F"], dtype="S"),
    )


def write_topology(h5f, topology, source):
    h5f["results/positions/number_ion_types"] = topology.number_ion_types
    h5f["results/positions/ion_types"] = topology.ion_types


def check_topology(file, reference, source):
    actual = get_actual_and_check_version(file, "topology", source)
    assert actual == reference


def test_cell(tmpdir):
    setup = SetupTest(
        directory=tmpdir,
        options=default_options,
        sources=default_sources,
        create_reference=reference_cell,
        write_reference=write_cell,
        check_actual=check_cell,
    )
    generic_test(setup)


def reference_cell():
    shape = (num_steps, 3, 3)
    return RawCell(
        scale=0.5,
        lattice_vectors=np.arange(np.prod(shape)).reshape(shape),
    )


def write_cell(h5f, cell, source):
    h5f["results/positions/scale"] = cell.scale
    h5f["/intermediate/ion_dynamics/lattice_vectors"] = cell.lattice_vectors


def check_cell(file, reference, source):
    actual = get_actual_and_check_version(file, "cell", source)
    assert actual == reference
    assert isinstance(actual.scale, Number)


def test_energies(tmpdir):
    setup = SetupTest(
        directory=tmpdir,
        options=default_options,
        sources=default_sources,
        create_reference=reference_energies,
        write_reference=write_energies,
        check_actual=check_energies,
    )
    generic_test(setup)


def reference_energies():
    labels = np.array(["total", "kinetic", "temperature"], dtype="S")
    shape = (100, len(labels))
    return RawEnergy(
        labels=labels,
        values=np.arange(np.prod(shape)).reshape(shape),
    )


def write_energies(h5f, energy, source):
    h5f["intermediate/ion_dynamics/energies_tags"] = energy.labels
    h5f["intermediate/ion_dynamics/energies"] = energy.values


def check_energies(file, reference, source):
    assert get_actual_and_check_version(file, "energy", source) == reference


def test_kpoints(tmpdir):
    setup = SetupTest(
        directory=tmpdir,
        options=itertools.product((True, False), repeat=2),
        sources=default_sources + ("kpoints_opt",),
        create_reference=reference_kpoints,
        write_reference=write_kpoints,
        check_actual=check_kpoints,
    )
    generic_test(setup)


def reference_kpoints(use_labels):
    kpoints = RawKpoints(
        mode="explicit",
        number=num_kpoints,
        coordinates=np.linspace(np.zeros(3), np.ones(3), num_kpoints),
        weights=np.arange(num_kpoints),
        cell=reference_cell(),
    )
    if use_labels:
        kpoints.labels = np.array(["G", "X"], dtype="S")
        kpoints.label_indices = [0, 1]
    return kpoints


def write_kpoints(h5f, kpoints, source):
    suffix = source_suffix(source)
    input = "input/kpoints" if suffix == "" else "input/kpoints_opt"
    h5f[f"{input}/mode"] = kpoints.mode
    h5f[f"{input}/number_kpoints"] = kpoints.number
    h5f[f"results/electron_eigenvalues{suffix}/kpoint_coords"] = kpoints.coordinates
    key = f"results/electron_eigenvalues{suffix}/kpoints_symmetry_weight"
    h5f[key] = kpoints.weights
    write_cell(h5f, kpoints.cell, source)
    if kpoints.label_indices is not None:
        h5f[f"{input}/positions_labels_kpoints"] = kpoints.label_indices
    if kpoints.labels is not None:
        h5f[f"{input}/labels_kpoints"] = kpoints.labels


def check_kpoints(file, reference, source):
    actual = get_actual_and_check_version(file, "kpoints", source)
    assert actual == reference
    assert isinstance(actual.number, Integral)
    assert isinstance(actual.mode, str)


def test_magnetism(tmpdir):
    setup = SetupTest(
        directory=tmpdir,
        options=default_options,
        sources=default_sources,
        create_reference=reference_magnetism,
        write_reference=write_magnetism,
        check_actual=check_magnetism,
    )
    generic_test(setup)


def reference_magnetism():
    shape = (num_steps, num_components, num_atoms, lmax)
    magnetism = RawMagnetism(moments=np.arange(np.prod(shape)).reshape(shape))
    return magnetism


def write_magnetism(h5f, magnetism, source):
    h5f["intermediate/ion_dynamics/magnetism/moments"] = magnetism.moments


def check_magnetism(file, reference, source):
    assert get_actual_and_check_version(file, "magnetism", source) == reference


def test_structure(tmpdir):
    setup = SetupTest(
        directory=tmpdir,
        options=itertools.product((True, False), repeat=2),
        sources=default_sources,
        create_reference=reference_structure,
        write_reference=write_structure,
        check_actual=check_structure,
    )
    generic_test(setup)


def reference_structure(use_magnetism):
    shape_pos = (num_steps, num_atoms, 3)
    structure = RawStructure(
        topology=reference_topology(),
        cell=reference_cell(),
        positions=np.arange(np.prod(shape_pos)).reshape(shape_pos),
        magnetism=reference_magnetism() if use_magnetism else None,
    )
    return structure


def write_structure(h5f, structure, source):
    write_topology(h5f, structure.topology, source)
    write_cell(h5f, structure.cell, source)
    h5f["intermediate/ion_dynamics/position_ions"] = structure.positions
    if structure.magnetism is not None:
        write_magnetism(h5f, structure.magnetism, source)


def check_structure(file, reference, source):
    assert get_actual_and_check_version(file, "structure", source) == reference


def test_density(tmpdir):
    setup = SetupTest(
        directory=tmpdir,
        options=default_options,
        sources=default_sources,
        create_reference=reference_density,
        write_reference=write_density,
        check_actual=check_density,
    )
    generic_test(setup)


def reference_density():
    shape = (2, 3, 4, 5)
    raw_data = np.arange(np.prod(shape)).reshape(shape)
    return RawDensity(
        structure=reference_structure(False),
        charge=raw_data,
    )


def write_density(h5f, density, source):
    write_structure(h5f, density.structure, source)
    h5f["charge/charge"] = density.charge


def check_density(file, reference, source):
    assert get_actual_and_check_version(file, "density", source) == reference


def test_dielectric(tmpdir):
    setup = SetupTest(
        directory=tmpdir,
        options=default_options,
        sources=default_sources + ("electron", "ion"),
        create_reference=reference_dielectric,
        write_reference=write_dielectric,
        check_actual=check_dielectric,
    )
    generic_test(setup)


def reference_dielectric():
    shape = (axes, axes, num_energies, complex)
    return RawDielectric(
        energies=np.arange(num_energies),
        function=np.arange(np.prod(shape)).reshape(shape),
    )


def write_dielectric(h5f, dielectric, source):
    group = source_prefix(source) + "dielectric"
    h5f[f"results/{group}/energies"] = dielectric.energies
    h5f[f"results/{group}/function"] = dielectric.function


def check_dielectric(file, reference, source):
    assert get_actual_and_check_version(file, "dielectric", source) == reference


def test_forces(tmpdir):
    setup = SetupTest(
        directory=tmpdir,
        options=default_options,
        sources=default_sources,
        create_reference=reference_forces,
        write_reference=write_forces,
        check_actual=check_forces,
    )
    generic_test(setup)


def reference_forces():
    shape = (num_steps, num_atoms, axes)
    return RawForces(
        structure=reference_structure(use_magnetism=False),
        forces=np.arange(np.prod(shape)).reshape(shape),
    )


def write_forces(h5f, forces, source):
    write_structure(h5f, forces.structure, source)
    h5f[f"intermediate/ion_dynamics/forces"] = forces.forces


def check_forces(file, reference, source):
    assert get_actual_and_check_version(file, "forces", source) == reference


def test_stress(tmpdir):
    setup = SetupTest(
        directory=tmpdir,
        options=default_options,
        sources=default_sources,
        create_reference=reference_stress,
        write_reference=write_stress,
        check_actual=check_stress,
    )
    generic_test(setup)


def reference_stress():
    shape = (num_steps, axes, axes)
    return RawStress(
        structure=reference_structure(use_magnetism=False),
        stress=np.arange(np.prod(shape)).reshape(shape),
    )


def write_stress(h5f, stress, source):
    write_structure(h5f, stress.structure, source)
    h5f[f"intermediate/ion_dynamics/stress"] = stress.stress


def check_stress(file, reference, source):
    assert get_actual_and_check_version(file, "stress", source) == reference


def source_prefix(source):
    return "electron_" if source == "default" else f"{source}_"


def source_suffix(source):
    return "" if source == "default" else f"_{source}"
