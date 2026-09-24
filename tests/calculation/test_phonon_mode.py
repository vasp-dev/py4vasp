# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import types

import h5py
import numpy as np
import pytest

import py4vasp
from py4vasp import raw
from py4vasp._calculation.phonon_mode import PhononMode, PhononModeHandler
from py4vasp._calculation.structure import Structure
from py4vasp._demo.phonon import mode as phonon_mode_demo
from py4vasp._raw.models import PhononModeModel
from py4vasp._util import masses


@pytest.fixture
def phonon_mode(raw_data):
    raw_mode = raw_data.phonon_mode("default")
    mode = PhononMode.from_data(raw_mode)
    mode.ref = types.SimpleNamespace()
    mode.ref.structure = Structure.from_data(raw_mode.structure)
    mode.ref.frequencies = raw_mode.frequencies.flatten().view(np.complex128)
    mode.ref.eigenvectors = raw_mode.eigenvectors
    mode.ref.raw_data = raw_mode
    return mode


def test_read(phonon_mode, Assert):
    actual = phonon_mode.read()
    Assert.same_structure(actual["structure"], phonon_mode.ref.structure.read())
    Assert.allclose(actual["frequencies"], phonon_mode.ref.frequencies)
    Assert.allclose(actual["eigenvectors"], phonon_mode.ref.eigenvectors)


def test_frequencies(phonon_mode, Assert):
    Assert.allclose(phonon_mode.frequencies(), phonon_mode.ref.frequencies)


def test_print(phonon_mode, format_):
    actual, _ = format_(phonon_mode)
    expected_text = """\
 Eigenvalues of the dynamical matrix
 -----------------------------------
   1 f  =   76.463537 THz   480.434572 2PiTHz 2550.569965 cm-1   316.227766 meV
   2 f  =   74.134150 THz   465.798600 2PiTHz 2472.869329 cm-1   306.594194 meV
   3 f  =   71.729156 THz   450.687578 2PiTHz 2392.646712 cm-1   296.647939 meV
   4 f  =   69.240678 THz   435.052008 2PiTHz 2309.639335 cm-1   286.356421 meV
   5 f  =   66.659366 THz   418.833150 2PiTHz 2223.535345 cm-1   275.680975 meV
   6 f  =   63.973985 THz   401.960402 2PiTHz 2133.959934 cm-1   264.575131 meV
   7 f  =   61.170830 THz   384.347658 2PiTHz 2040.455972 cm-1   252.982213 meV
   8 f  =   58.232895 THz   365.888069 2PiTHz 1942.456214 cm-1   240.831892 meV
   9 f  =   55.138641 THz   346.446297 2PiTHz 1839.242158 cm-1   228.035085 meV
  10 f  =   51.860094 THz   325.846580 2PiTHz 1729.880715 cm-1   214.476106 meV
  11 f  =   48.359787 THz   303.853503 2PiTHz 1613.122084 cm-1   200.000000 meV
  12 f  =   44.585521 THz   280.139088 2PiTHz 1487.225077 cm-1   184.390889 meV
  13 f  =   40.460701 THz   254.222080 2PiTHz 1349.634766 cm-1   167.332005 meV
  14 f  =   35.864578 THz   225.343789 2PiTHz 1196.323356 cm-1   148.323970 meV
  15 f  =   30.585415 THz   192.173829 2PiTHz 1020.227986 cm-1   126.491106 meV
  16 f  =   24.179893 THz   151.926751 2PiTHz  806.561042 cm-1   100.000000 meV
  17 f  =   15.292707 THz    96.086914 2PiTHz  510.113993 cm-1    63.245553 meV
  18 f/i=   10.813577 THz    67.943709 2PiTHz  360.705064 cm-1    44.721360 meV
  19 f/i=   21.627154 THz   135.887418 2PiTHz  721.410127 cm-1    89.442719 meV
  20 f/i=   28.610036 THz   179.762157 2PiTHz  954.335895 cm-1   118.321596 meV
  21 f/i=   34.195533 THz   214.856872 2PiTHz 1140.649564 cm-1   141.421356 meV
"""
    assert actual == {"text/plain": expected_text}


def test_to_database(phonon_mode):
    handler = PhononModeHandler.from_data(phonon_mode.ref.raw_data)
    db_data: PhononModeModel = handler.to_database()
    assert isinstance(db_data, PhononModeModel)
    assert db_data.frequencies_real_max == float(
        np.max(phonon_mode.ref.frequencies.real)
    )
    assert db_data.frequencies_imag_max == float(
        np.max(phonon_mode.ref.frequencies.imag)
    )


def test_to_database_dispatch(phonon_mode):
    """The standalone dispatcher keys the result by the full quantity name."""
    result = phonon_mode._to_database()
    assert set(result) == {"phonon_mode"}


def test_to_database_in_calculation(tmp_path):
    """Collecting a group member at the calculation level keys it ``<group>_<member>``
    (``phonon_mode``), not the doubled ``phonon_phonon_mode``."""
    with h5py.File(tmp_path / "vaspout.h5", "w") as h5f:
        py4vasp._raw.write.write(h5f, raw.Version(99, 99, 99))
        py4vasp._raw.write.write(h5f, phonon_mode_demo.Sr2TiO4())
    properties = py4vasp.Calculation.from_path(tmp_path)._to_database().properties
    assert "phonon_mode" in properties
    assert "phonon_phonon_mode" not in properties


def test_print_writes_to_stdout(phonon_mode, capsys):
    assert phonon_mode.print() is None
    assert capsys.readouterr().out == str(phonon_mode) + "\n"


def test_selections(phonon_mode):
    assert phonon_mode.selections() == {"phonon_mode": ["default"]}


@pytest.fixture
def translation_mode(raw_data):
    """A phonon mode whose first eigenvector moves every atom by the same amount.

    VASP weights the eigenvectors with the square root of the mass, so translating the
    whole crystal along x does not give every atom the same eigenvector but one that is
    proportional to sqrt(mass). Undoing that weighting has to bring back the uniform
    displacement, which is what makes this pattern a useful reference.
    """
    raw_mode = raw_data.phonon_mode("default")
    number_modes = len(raw_mode.eigenvectors)
    elements = Structure.from_data(raw_mode.structure).read()["elements"]
    mass = masses.of(elements)
    translation = np.zeros((len(mass), 3))
    translation[:, 0] = np.sqrt(mass / np.sum(mass))
    eigenvectors = np.eye(number_modes)
    eigenvectors[0] = translation.flatten()
    raw_mode = dataclasses.replace(raw_mode, eigenvectors=eigenvectors)
    mode = PhononModeHandler.from_data(raw_mode)
    mode.ref = types.SimpleNamespace()
    mode.ref.masses = mass
    mode.ref.uniform_displacement = 1 / np.sqrt(np.sum(mass))
    mode.ref.eigenvectors = eigenvectors
    return mode


def test_displacements_undo_the_mass_weighting(translation_mode, Assert):
    actual = translation_mode.displacements()[0]
    expected = np.zeros_like(actual)
    expected[:, 0] = translation_mode.ref.uniform_displacement
    Assert.allclose(actual, expected)


def test_displacements_are_normalized_to_unit_normal_coordinate(
    translation_mode, Assert
):
    # the normal coordinate Q^2 = sum_i m_i u_i^2 is what sets the energy of a mode, so
    # every pattern is scaled to Q = 1 and displace only multiplies the physical scale
    displacements = translation_mode.displacements()
    mass = translation_mode.ref.masses[:, np.newaxis]
    normal_coordinate = np.sum(mass * displacements**2, axis=(1, 2))
    Assert.allclose(normal_coordinate, np.ones(len(displacements)))


def test_displacements_read_both_eigenvector_shapes(translation_mode, Assert):
    # VASP writes the eigenvectors as (mode, atom, direction) whereas the demo data
    # flattens the two trailing axes; both describe the same displacement
    flat = translation_mode.ref.eigenvectors
    raw_mode = dataclasses.replace(
        translation_mode._raw_phonon_mode,
        eigenvectors=flat.reshape(len(flat), -1, 3),
    )
    nested = PhononModeHandler.from_data(raw_mode)
    Assert.allclose(nested.displacements(), translation_mode.displacements())


def test_displacements_accept_custom_masses(translation_mode, Assert):
    # with all masses equal to one, undoing the weighting leaves the eigenvectors
    number_atoms = len(translation_mode.ref.masses)
    actual = translation_mode.displacements(masses=np.ones(number_atoms))
    expected = translation_mode.ref.eigenvectors.reshape(-1, number_atoms, 3)
    Assert.allclose(actual, expected)


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.phonon_mode("Sr2TiO4")
    check_factory_methods(PhononMode, data, skip_methods=["selections"])
