# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest
import types
from py4vasp.data import PhononDos


@pytest.fixture
def phonon_dos(raw_data):
    raw_dos = raw_data.phonon_dos("default")
    dos = PhononDos.from_data(raw_dos)
    dos.ref = types.SimpleNamespace()
    return dos


def test_phonon_dos_read(phonon_dos, Assert):
    pass
