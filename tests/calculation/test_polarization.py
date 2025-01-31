# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp._calculation.polarization import Polarization


@pytest.fixture
def polarization(raw_data):
    raw_polarization = raw_data.polarization("default")
    polarization = Polarization.from_data(raw_polarization)
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


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.polarization("default")
    check_factory_methods(Polarization, data)
