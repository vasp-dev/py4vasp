from py4vasp.data import Dos
import pytest
import h5py
import numpy as np
from tempfile import TemporaryFile


@pytest.fixture
def nonmagnetic_Dos():
    """ Setup a nonmagnetic Dos file containing all important quantities."""
    tf = TemporaryFile()
    h5f = h5py.File(tf, "a")
    energies = np.linspace(-1, 3)
    h5f["results/dos/energies"] = energies
    h5f["results/dos/dos"] = np.array([energies ** 2])
    h5f["results/dos/efermi"] = 1.372
    return h5f


def test_nonmagnetic_Dos_read(nonmagnetic_Dos):
    """ Test whether reading the nonmagnetic Dos yields the expected results."""
    dos = Dos(nonmagnetic_Dos).read()
    fermi_energy = nonmagnetic_Dos["results/dos/efermi"][()]
    energies = nonmagnetic_Dos["results/dos/energies"][:] - fermi_energy
    assert np.all(dos.energies == energies)
    assert np.all(dos.total == nonmagnetic_Dos["results/dos/dos"][0, :])
    assert dos.fermi_energy == fermi_energy


def test_nonmagnetic_Dos_plot(nonmagnetic_Dos):
    """ Test whether plotting the nonmagnetic Dos yields the expected results."""
    fig = Dos(nonmagnetic_Dos).plot()
    fermi_energy = nonmagnetic_Dos["results/dos/efermi"][()]
    energies = nonmagnetic_Dos["results/dos/energies"][:] - fermi_energy
    assert fig.layout.xaxis.title.text == "Energy (eV)"
    assert fig.layout.yaxis.title.text == "DOS (1/eV)"
    assert np.all(fig.data[0].x == energies)
    assert np.all(fig.data[0].y == nonmagnetic_Dos["results/dos/dos"][0, :])


@pytest.fixture
def magnetic_Dos():
    """ Setup a magnetic Dos file containing all relevant quantities."""
    tf = TemporaryFile()
    h5f = h5py.File(tf, "a")
    energies = np.linspace(-2, 2)
    h5f["results/dos/energies"] = energies
    h5f["results/dos/dos"] = np.array([(energies + 0.5) ** 2, (energies - 0.5) ** 2])
    h5f["results/dos/efermi"] = -0.137
    return h5f


def test_magnetic_Dos_read(magnetic_Dos):
    """ Test whether plotting the magnetic Dos yields the expected results."""
    dos = Dos(magnetic_Dos).read()
    fermi_energy = magnetic_Dos["results/dos/efermi"][()]
    energies = magnetic_Dos["results/dos/energies"][:] - fermi_energy
    assert np.all(dos.energies == energies)
    assert np.all(dos.up == magnetic_Dos["results/dos/dos"][0, :])
    assert np.all(dos.down == magnetic_Dos["results/dos/dos"][1, :])
    assert dos.fermi_energy == fermi_energy


def test_magnetic_Dos_plot(magnetic_Dos):
    """ Test whether plotting the magnetic Dos yields the expected results."""
    fig = Dos(magnetic_Dos).plot()
    assert np.all(fig.data[0].x == fig.data[1].x)
    assert np.all(fig.data[0].y == magnetic_Dos["results/dos/dos"][0, :])
    assert np.all(fig.data[1].y == -magnetic_Dos["results/dos/dos"][1, :])
