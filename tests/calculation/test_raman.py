# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp import exception, raw
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


def make_raman(tensors, energies=(0.0, 1.0, 2.0)):
    """Build a Raman quantity whose tensor is the given one at every photon energy."""
    tensors = np.asarray(tensors, dtype=np.complex128)
    shape = tensors.shape + (len(energies),)
    resolved = np.broadcast_to(tensors[..., np.newaxis], shape)
    return Raman.from_data(
        raw.Raman(
            frequencies=raw.VaspData(np.linspace(100, 500, len(tensors))),
            energies=raw.VaspData(np.array(energies, dtype=np.float64)),
            raman_tensor=raw.VaspData(
                np.stack((resolved.real, resolved.imag), axis=-1)
            ),
        )
    )


def invariants(tensor):
    """The two rotational invariants, written out rather than taken from the code."""
    mean = (tensor[0, 0] + tensor[1, 1] + tensor[2, 2]) / 3
    isotropic = abs(mean) ** 2
    anisotropy = 0.5 * (
        abs(tensor[0, 0] - tensor[1, 1]) ** 2
        + abs(tensor[1, 1] - tensor[2, 2]) ** 2
        + abs(tensor[2, 2] - tensor[0, 0]) ** 2
    ) + 3 * (abs(tensor[0, 1]) ** 2 + abs(tensor[1, 2]) ** 2 + abs(tensor[0, 2]) ** 2)
    return isotropic, anisotropy


def test_activity_returns_the_frequencies_and_the_laser(raman, Assert):
    actual = raman.activity()
    assert sorted(actual) == ["frequencies", "laser", "powder"]
    Assert.allclose(actual["frequencies"], raman.read()["frequencies"])


def test_powder_activity_matches_the_closed_form(raman, Assert):
    tensors = raman.read()["raman_tensor"][..., 0]
    expected = [45 * a2 + 7 * g2 for a2, g2 in map(invariants, tensors)]
    Assert.allclose(raman.activity()["powder"], np.array(expected))


def test_isotropic_and_anisotropy_match_the_closed_form(raman, Assert):
    tensors = raman.read()["raman_tensor"][..., 0]
    expected = np.array(list(map(invariants, tensors)))
    actual = raman.activity("isotropic, anisotropy")
    Assert.allclose(actual["isotropic"], expected[:, 0])
    Assert.allclose(actual["anisotropy"], expected[:, 1])


def test_oriented_component_is_the_squared_element(raman, Assert):
    tensors = raman.read()["raman_tensor"][..., 0]
    actual = raman.activity("xx, xy, zz")
    Assert.allclose(actual["xx"], np.abs(tensors[:, 0, 0]) ** 2)
    Assert.allclose(actual["xy"], np.abs(tensors[:, 0, 1]) ** 2)
    Assert.allclose(actual["zz"], np.abs(tensors[:, 2, 2]) ** 2)


def test_depolarization_ratio_of_an_isotropic_tensor_vanishes(Assert):
    # a mode that polarizes the crystal equally in every direction scatters light that
    # keeps the polarization of the laser
    raman = make_raman([np.eye(3), 2.5 * np.eye(3)])
    Assert.allclose(raman.activity("depolarization")["depolarization"], np.zeros(2))


def test_depolarization_ratio_of_a_traceless_tensor_is_three_quarters(Assert):
    # the other analytic limit: with no isotropic part the ratio saturates at 3/4,
    # which is how an experiment recognizes a mode that is not totally symmetric
    traceless = np.diag((1.0, -1.0, 0.0))
    shear = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    raman = make_raman([traceless, shear])
    actual = raman.activity("depolarization")["depolarization"]
    Assert.allclose(actual, np.full(2, 0.75))


def test_polarized_and_depolarized_add_up_to_the_powder_average(raman, Assert):
    # 45a^2 + 4g^2 and 3g^2 are the two halves of 45a^2 + 7g^2
    actual = raman.activity("parallel, perpendicular, powder")
    total = actual["parallel"] + actual["perpendicular"]
    Assert.allclose(total, actual["powder"])


def test_laser_snaps_to_the_nearest_grid_point(raman, Assert):
    energies = raman.read()["energies"]
    spacing = energies[1] - energies[0]
    actual = raman.activity(laser=energies[3] + 0.4 * spacing)
    Assert.allclose(actual["laser"], energies[3])
    Assert.allclose(actual["powder"], raman.activity(laser=energies[3])["powder"])


def test_laser_selects_the_tensor_at_that_photon_energy(raman, Assert):
    energies = raman.read()["energies"]
    tensors = raman.read()["raman_tensor"][..., 3]
    expected = [45 * a2 + 7 * g2 for a2, g2 in map(invariants, tensors)]
    actual = raman.activity(laser=energies[3])
    Assert.allclose(actual["powder"], np.array(expected))


def test_laser_outside_the_grid_raises_error(raman):
    maximum = raman.read()["energies"].max()
    with pytest.raises(exception.IncorrectUsage):
        raman.activity(laser=maximum + 1.0)
    with pytest.raises(exception.IncorrectUsage):
        raman.activity(laser=-1.0)


def test_unknown_selection_raises_error(raman):
    with pytest.raises(exception.IncorrectUsage):
        raman.activity("not_an_observable")


def test_tensor_that_is_not_symmetric_raises_error():
    # the invariants assume the symmetry VASP writes; silently averaging an asymmetric
    # tensor would report a plausible number for data py4vasp does not understand
    asymmetric = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    raman = make_raman([asymmetric])
    with pytest.raises(exception.DataMismatch):
        raman.activity()


@pytest.fixture
def two_modes():
    """A totally symmetric mode and a traceless one, with invariants known by hand.

    The identity has a mean polarizability of 1 and no anisotropy, so its powder
    activity is 45 and it scatters no depolarized light at all. The traceless tensor
    has no isotropic part and an anisotropy of 3, so its activity is 7 * 3 and its
    depolarization ratio sits at the maximum of 3/4.
    """
    return make_raman([np.eye(3), np.diag((1.0, -1.0, 0.0))])


def test_print(two_modes, format_):
    actual, _ = format_(two_modes)
    expected_text = """\
Raman activity at a laser energy of 0.00 eV
-------------------------------------------
mode   omega (cm-1)   omega (meV)      activity   depolarization
   1         100.00         12.40       45.0000           0.0000
   2         500.00         61.99       21.0000           0.7500"""
    assert actual == {"text/plain": expected_text}


def test_print_reports_the_modes_in_the_order_vasp_wrote_them(raman, Assert):
    # VASP writes the frequencies in descending order and the table follows it, so a
    # row can be compared against the OUTCAR line by line
    rows = str(raman).splitlines()[3:]
    printed = [float(row.split()[1]) for row in rows]
    Assert.allclose(np.array(printed), raman.read()["frequencies"] * convert.EV_TO_CM1)


def test_print_writes_to_stdout(raman, capsys):
    assert raman.print() is None
    assert capsys.readouterr().out == str(raman) + "\n"
