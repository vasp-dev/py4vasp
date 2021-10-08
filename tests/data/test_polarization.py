import pytest
import types
from py4vasp.data import Polarization


@pytest.fixture
def polarization(raw_data):
    raw_polarization = raw_data.polarization("default")
    polarization = Polarization(raw_polarization)
    polarization.ref = types.SimpleNamespace()
    polarization.ref.ion_dipole = raw_polarization.ion
    polarization.ref.electron_dipole = raw_polarization.electron
    return polarization


def test_read(polarization, Assert):
    actual = polarization.read()
    Assert.allclose(actual["ion_dipole"], polarization.ref.ion_dipole)
    Assert.allclose(actual["electron_dipole"], polarization.ref.electron_dipole)


def test_print(polarization, format_):
    actual, _ = format_(polarization)
    reference = f"""
Polarization (|e|Å)
-------------------------------------------------------------
ionic dipole moment:          4.00000     5.00000     6.00000
electronic dipole moment:     1.00000     2.00000     3.00000
""".strip()
    assert actual == {"text/plain": reference}


def test_descriptor(polarization, check_descriptors):
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_to_string": ["__str__"],
    }
    check_descriptors(polarization, descriptors)


def test_from_file(raw_data, mock_file, check_read):
    raw_polarization = raw_data.polarization("default")
    with mock_file("polarization", raw_polarization) as mocks:
        check_read(Polarization, mocks, raw_polarization)
