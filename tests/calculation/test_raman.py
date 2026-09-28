# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation.raman import Raman
from py4vasp._util import convert


@pytest.fixture
def raman(raw_data):
    raw_raman = raw_data.raman("Sr2TiO4")
    raman = Raman.from_data(raw_raman)
    raman.ref = types.SimpleNamespace()
    raman.ref.raw = raw_raman
    raman.ref.frequencies = np.array(raw_raman.frequencies) / convert.EV_TO_CM1
    raman.ref.energies = np.array(raw_raman.energies)
    raman.ref.raman_tensor = convert.to_complex(np.array(raw_raman.raman_tensor))
    # a mode of exactly zero frequency is not a vibration, so it is dropped whatever
    # threshold is passed; comparing against every mode would never be right
    raman.ref.vibrating = raman.ref.frequencies > 0
    return raman


def test_read(raman, Assert):
    actual = raman.read(minimum_frequency=0)
    vibrating = raman.ref.vibrating
    Assert.allclose(actual["frequencies"], raman.ref.frequencies[vibrating])
    Assert.allclose(actual["energies"], raman.ref.energies)
    Assert.allclose(actual["raman_tensor"], raman.ref.raman_tensor[vibrating])


def test_to_dict_is_alias_of_read(raman, Assert):
    Assert.allclose(raman.to_dict()["frequencies"], raman.read()["frequencies"])


def test_read_converts_frequencies_to_eV(raman, Assert):
    # VASP stores cm^-1 here, but every value py4vasp returns is an energy in eV
    raw_frequencies = np.array(raman.ref.raw.frequencies)[raman.ref.vibrating]
    actual = raman.read(minimum_frequency=0)["frequencies"]
    Assert.allclose(actual * convert.EV_TO_CM1, raw_frequencies)


def test_read_returns_a_complex_tensor(raman):
    tensor = raman.read()["raman_tensor"]
    number_modes = len(raman.read()["frequencies"])
    number_energies = len(raman.ref.energies)
    assert tensor.shape == (number_modes, 3, 3, number_energies)
    assert np.iscomplexobj(tensor)


def test_read_drops_the_modes_without_a_frequency(raman, Assert):
    # the lowest modes translate or rotate the system instead of vibrating it
    minimum_frequency = 0.01  # eV
    kept = raman.ref.frequencies > minimum_frequency
    actual = raman.read(minimum_frequency=minimum_frequency)
    assert np.any(kept) and not np.all(kept)
    Assert.allclose(actual["frequencies"], raman.ref.frequencies[kept])
    Assert.allclose(actual["raman_tensor"], raman.ref.raman_tensor[kept])


def test_negative_minimum_frequency_raises_error(raman):
    with pytest.raises(exception.IncorrectUsage):
        raman.read(minimum_frequency=-1.0)


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.raman("Sr2TiO4")
    # selections reports what the schema offers rather than reading the file, exactly
    # as it does for the dielectric tensor
    check_factory_methods(Raman, data, skip_methods=["selections"])
