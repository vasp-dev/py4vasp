import py4vasp.raw as raw
from py4vasp.raw import File
import contextlib
import pytest
import h5py
import os
import numpy as np
import itertools
import inspect
from tempfile import TemporaryFile
from collections import namedtuple
from numbers import Number, Integral

num_spins = 2
num_energies = 20
num_kpoints = 10
num_bands = 3
num_atoms = 10  # sum(range(5))
lmax = 3
fermi_energy = 0.123

SetupTest = namedtuple(
    "SetupTest", "directory, options, create_reference, write_reference, check_actual"
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


def test_file_as_context():
    tf = TemporaryFile()
    h5f = h5py.File(tf, "w")
    h5f.close()
    with File(tf) as file:
        assert not file.closed
        h5f = file._h5f
    # check that file is closed and accessing it raises ValueError
    assert file.closed
    with pytest.raises(ValueError):
        h5f.file
    for func in inspect.getmembers(file, predicate=inspect.isroutine):
        name = func[0]
        if name[0] == "_" or name in ["close"]:
            continue
        with pytest.raises(AssertionError):
            getattr(file, name)()


def generic_test(setup):
    with working_directory(setup.directory):
        for option in setup.options:
            use_default, *additional_tags = option
            reference = setup.create_reference(*additional_tags)
            h5f, filename = open_h5_file(use_default)
            setup.write_reference(h5f, reference)
            h5f.close()
            file = open_vasp_file(use_default, filename)
            setup.check_actual(file, reference)
            file.close()  # must be after comparison, because file is read lazily


def open_h5_file(use_default):
    filename = "vaspout.h5" if use_default else TemporaryFile()
    return h5py.File(filename, "w"), filename


def open_vasp_file(use_default, filename):
    return File() if use_default else File(filename)


def test_dos(tmpdir):
    setup = SetupTest(
        directory=tmpdir,
        options=itertools.product((True, False), repeat=2),
        create_reference=reference_dos,
        write_reference=write_dos,
        check_actual=check_dos,
    )
    generic_test(setup)


def reference_dos(use_projectors):
    shape = (num_spins, num_energies)
    return raw.Dos(
        fermi_energy=fermi_energy,
        energies=np.arange(num_energies),
        dos=np.arange(np.prod(shape)).reshape(shape),
        projectors=reference_projectors() if use_projectors else None,
    )


def write_dos(h5f, dos):
    h5f["results/electron_dos/efermi"] = dos.fermi_energy
    h5f["results/electron_dos/energies"] = dos.energies
    h5f["results/electron_dos/dos"] = dos.dos
    if dos.projectors is not None:
        write_projectors(h5f, dos.projectors)
    if dos.projections is not None:
        h5f["results/electron_dos/dospar"] = proj.dos


def check_dos(file, reference):
    actual = file.dos()
    assert actual == reference
    assert isinstance(actual.fermi_energy, Number)


def test_band(tmpdir):
    setup = SetupTest(
        directory=tmpdir,
        options=itertools.product((True, False), repeat=3),
        create_reference=reference_band,
        write_reference=write_band,
        check_actual=check_band,
    )
    generic_test(setup)


def reference_band(use_projectors, use_labels):
    shape_eval = (num_spins, num_kpoints, num_bands)
    shape_proj = (num_spins, num_atoms, lmax, num_kpoints, num_bands)
    band = raw.Band(
        fermi_energy=fermi_energy,
        kpoints=reference_kpoints(use_labels),
        eigenvalues=np.arange(np.prod(shape_eval)).reshape(shape_eval),
    )
    if use_projectors:
        band.projectors = reference_projectors()
        band.projections = np.arange(np.prod(shape_proj)).reshape(shape_proj)
    return band


def write_band(h5f, band):
    h5f["results/electron_dos/efermi"] = band.fermi_energy
    h5f["results/electron_eigenvalues/eigenvalues"] = band.eigenvalues
    write_kpoints(h5f, band.kpoints)
    if band.projectors is not None:
        write_projectors(h5f, band.projectors)
    if band.projections is not None:
        h5f["results/projectors/par"] = band.projections


def check_band(file, reference):
    actual = file.band()
    assert actual == reference
    assert isinstance(actual.fermi_energy, Number)


def test_projectors(tmpdir):
    setup = SetupTest(
        directory=tmpdir,
        options=((True,), (False,)),
        create_reference=reference_projectors,
        write_reference=write_projectors,
        check_actual=check_projectors,
    )
    generic_test(setup)


def reference_projectors():
    shape_dos = (num_spins, num_atoms, lmax, num_energies)
    return raw.Projectors(
        number_ion_types=np.arange(5),
        ion_types=np.array(["B", "C", "N", "O", "F"], dtype="S"),
        orbital_types=np.array(["s", "p", "d", "f"], dtype="S"),
        number_spins=num_spins,
    )


def write_projectors(h5f, proj):
    h5f["results/positions/number_ion_types"] = proj.number_ion_types
    h5f["results/positions/ion_types"] = proj.ion_types
    h5f["results/projectors/lchar"] = proj.orbital_types
    h5f["results/electron_eigenvalues/ispin"] = proj.number_spins


def check_projectors(file, reference):
    actual = file.projectors()
    assert actual == reference
    assert isinstance(actual.number_spins, Integral)


def test_cell(tmpdir):
    setup = SetupTest(
        directory=tmpdir,
        options=((True,), (False,)),
        create_reference=reference_cell,
        write_reference=write_cell,
        check_actual=check_cell,
    )
    generic_test(setup)


def reference_cell():
    return raw.Cell(scale=0.5, lattice_vectors=np.arange(9).reshape(3, 3))


def write_cell(h5f, cell):
    h5f["results/positions/scale"] = cell.scale
    h5f["results/positions/lattice_vectors"] = cell.lattice_vectors


def check_cell(file, reference):
    actual = file.cell()
    assert actual == reference
    assert isinstance(actual.scale, Number)


def test_energies(tmpdir):
    setup = SetupTest(
        directory=tmpdir,
        options=((True,), (False,)),
        create_reference=reference_energies,
        write_reference=write_energies,
        check_actual=check_energies,
    )
    generic_test(setup)


def reference_energies():
    labels = np.array(["total", "kinetic", "temperature"], dtype="S")
    shape = (100, len(labels))
    return raw.Convergence(
        labels=labels, energies=np.arange(np.prod(shape)).reshape(shape)
    )


def write_energies(h5f, convergence):
    h5f["intermediate/history/energies_tags"] = convergence.labels
    h5f["intermediate/history/energies"] = convergence.energies


def check_energies(file, reference):
    assert file.convergence() == reference


def test_kpoints(tmpdir):
    setup = SetupTest(
        directory=tmpdir,
        options=itertools.product((True, False), repeat=2),
        create_reference=reference_kpoints,
        write_reference=write_kpoints,
        check_actual=check_kpoints,
    )
    generic_test(setup)


def reference_kpoints(use_labels):
    kpoints = raw.Kpoints(
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


def write_kpoints(h5f, kpoints):
    h5f["input/kpoints/mode"] = kpoints.mode
    h5f["input/kpoints/number_kpoints"] = kpoints.number
    h5f["results/electron_eigenvalues/kpoint_coords"] = kpoints.coordinates
    h5f["results/electron_eigenvalues/kpoints_symmetry_weight"] = kpoints.weights
    write_cell(h5f, kpoints.cell)
    if kpoints.label_indices is not None:
        h5f["input/kpoints/positions_labels_kpoints"] = kpoints.label_indices
    if kpoints.labels is not None:
        h5f["input/kpoints/labels_kpoints"] = kpoints.labels


def check_kpoints(file, reference):
    actual = file.kpoints()
    assert actual == reference
    assert isinstance(actual.number, Integral)
    assert isinstance(actual.mode, str)
