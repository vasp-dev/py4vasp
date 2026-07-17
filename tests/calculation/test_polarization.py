# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp._calculation.polarization import Polarization, PolarizationHandler
from py4vasp._raw.models import PolarizationModel


@pytest.fixture
def polarization(raw_data):
    raw_polarization = raw_data.polarization("default")
    polarization = Polarization.from_data(raw_polarization)
    polarization.ref = types.SimpleNamespace()
    polarization.ref.ion_dipole = raw_polarization.ion
    polarization.ref.electron_dipole = raw_polarization.electron
    return polarization


@pytest.fixture
def polarization_handler(raw_data):
    raw_polarization = raw_data.polarization("default")
    return PolarizationHandler.from_data(raw_polarization)


def test_read(polarization, Assert):
    actual = polarization.read()
    Assert.allclose(actual["ion_dipole"], polarization.ref.ion_dipole)
    Assert.allclose(actual["electron_dipole"], polarization.ref.electron_dipole)


def test_to_dict_matches_read(polarization_handler, Assert):
    result_to_dict = polarization_handler.to_dict()
    result_read = polarization_handler.to_dict()
    assert result_to_dict.keys() == result_read.keys()
    for key in result_read:
        Assert.allclose(result_to_dict[key], result_read[key])


def test_dispatcher_to_dict_matches_read(polarization, Assert):
    result_to_dict = polarization.to_dict()
    result_read = polarization.read()
    assert result_to_dict.keys() == result_read.keys()
    for key in result_read:
        Assert.allclose(result_to_dict[key], result_read[key])


def test_print(polarization, format_):
    actual, _ = format_(polarization)
    reference = f"""
Polarization (|e|Å)
-------------------------------------------------------------
ionic dipole moment:          4.00000     5.00000     6.00000
electronic dipole moment:     1.00000     2.00000     3.00000
""".strip()
    assert actual == {"text/plain": reference}


def test_to_database(raw_data):
    raw_polarization = raw_data.polarization("default")
    handler = PolarizationHandler.from_data(raw_polarization)
    db_data: PolarizationModel = handler.to_database()
    assert isinstance(db_data, PolarizationModel)

    assert db_data.ionic_dipole_moment == list(raw_polarization.ion[:])
    assert db_data.electronic_dipole_moment == list(raw_polarization.electron[:])
    total_dipole = raw_polarization.ion[:] + raw_polarization.electron[:]
    assert db_data.total_dipole_moment == list(total_dipole)
    assert db_data.ionic_dipole_norm == float(np.linalg.norm(raw_polarization.ion[:]))
    assert db_data.electronic_dipole_norm == float(
        np.linalg.norm(raw_polarization.electron[:])
    )
    assert db_data.total_dipole_norm == float(np.linalg.norm(total_dipole))


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.polarization("default")
    check_factory_methods(Polarization, data)
