# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp._calculation.phonon_mode import PhononMode
from py4vasp._calculation.structure import Structure


@pytest.fixture
def phonon_mode(raw_data):
    raw_mode = raw_data.phonon_mode("default")
    mode = PhononMode.from_data(raw_mode)
    mode.ref = types.SimpleNamespace()
    mode.ref.structure = Structure.from_data(raw_mode.structure)
    mode.ref.frequencies = raw_mode.frequencies
    mode.ref.eigenvectors = raw_mode.eigenvectors
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


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.phonon_mode("Sr2TiO4")
    check_factory_methods(PhononMode, data)
