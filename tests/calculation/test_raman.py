# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp import broadening, exception, raw
from py4vasp._calculation.raman import Raman, RamanHandler
from py4vasp._raw.models import RamanModel
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
    # as it does for the dielectric tensor. intensity has no default laser energy,
    # because a measured intensity is not defined without one.
    parameters = {"intensity": {"laser": 0.5}}
    check_factory_methods(Raman, data, parameters, skip_methods=["selections"])


def make_raman(tensors, energies=(0.0, 1.0, 2.0), frequencies=None):
    """Build a Raman quantity whose tensor is the given one at every photon energy."""
    tensors = np.asarray(tensors, dtype=np.complex128)
    shape = tensors.shape + (len(energies),)
    resolved = np.broadcast_to(tensors[..., np.newaxis], shape)
    if frequencies is None:
        frequencies = np.linspace(100, 500, len(tensors))
    return Raman.from_data(
        raw.Raman(
            frequencies=raw.VaspData(np.array(frequencies, dtype=np.float64)),
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


def test_print_reports_the_modes_in_the_order_vasp_wrote_them(raman, Assert):
    # VASP writes the frequencies in descending order and the table follows it, so a
    # row can be compared against the OUTCAR line by line
    rows = str(raman).splitlines()[3:]
    printed = [float(row.split()[2]) for row in rows]
    Assert.allclose(np.array(printed), raman.read()["frequencies"] * convert.EV_TO_CM1)


def test_print_writes_to_stdout(raman, capsys):
    assert raman.print() is None
    assert capsys.readouterr().out == str(raman) + "\n"


def test_to_graph_default(raman):
    graph = raman.to_graph()
    assert len(graph.series) == 1
    assert graph.series[0].label == "powder"
    assert graph.xlabel == "ω (meV)"
    assert "activity" in graph.ylabel


def test_to_graph_puts_the_modes_on_the_axis(raman, Assert):
    graph = raman.to_graph()
    frequencies = raman.read()["frequencies"] * convert.EV_TO_MEV
    mesh = graph.series[0].x
    # every line has to be on the axis, with room for its flanks on both sides
    assert mesh[0] < np.min(frequencies) and mesh[-1] > np.max(frequencies)
    assert np.all(np.diff(mesh) > 0)


def test_to_graph_conserves_the_total_activity(raman, Assert):
    # the line shapes of py4vasp.broadening carry unit area, so broadening moves the
    # activity around the axis without creating or destroying any of it. A Gaussian
    # falls off fast enough that the axis holds all of it.
    graph = raman.to_graph(shape=broadening.Gaussian(fwhm=1e-3))
    series = graph.series[0]
    total = np.sum(raman.activity()["powder"])
    Assert.allclose(np.trapezoid(series.y, series.x), total)


def test_to_graph_with_a_lorentzian_keeps_most_of_the_activity(raman):
    # a Lorentzian has tails that reach beyond any window, so no finite axis can hold
    # all of its area. It must hold nearly all of it and never more.
    graph = raman.to_graph()
    series = graph.series[0]
    total = np.sum(raman.activity()["powder"])
    integral = np.trapezoid(series.y, series.x)
    assert 0.85 * total < integral < total


def test_to_graph_with_several_selections(raman):
    graph = raman.to_graph("parallel, perpendicular")
    assert [series.label for series in graph.series] == ["parallel", "perpendicular"]


def test_to_graph_accepts_a_line_shape(raman, Assert):
    narrow = raman.to_graph(shape=broadening.Gaussian(fwhm=5e-4))
    wide = raman.to_graph(shape=broadening.Gaussian(fwhm=2e-3))
    # a wider line is a lower one, because both carry the same area
    assert np.max(narrow.series[0].y) > np.max(wide.series[0].y)
    # the two spectra live on meshes of different spacing, so the trapezoidal rule
    # gives their common area to its own accuracy rather than to machine precision
    Assert.allclose(
        np.trapezoid(narrow.series[0].y, narrow.series[0].x),
        np.trapezoid(wide.series[0].y, wide.series[0].x),
        tolerance=1e6,
    )


def test_to_graph_with_a_gaussian(raman):
    graph = raman.to_graph(shape=broadening.Gaussian(fwhm=1e-3))
    assert np.all(graph.series[0].y >= 0)


def test_to_graph_at_a_laser_energy(raman, Assert):
    laser = raman.read()["energies"][3]
    graph = raman.to_graph(laser=laser, shape=broadening.Gaussian(fwhm=1e-3))
    total = np.sum(raman.activity(laser=laser)["powder"])
    Assert.allclose(np.trapezoid(graph.series[0].y, graph.series[0].x), total)


def test_to_graph_rejects_something_that_is_not_a_line_shape(raman):
    with pytest.raises(exception.IncorrectUsage):
        raman.to_graph(shape=1e-3)


def test_selections(raman):
    selections = raman.selections()
    assert selections["raman"] == ["default"]
    assert "powder" in selections["observables"]
    assert "depolarization" in selections["observables"]
    assert selections["directions"][0] == "xx"


def test_print_reports_the_degeneracy_of_every_level():
    # two modes at the same frequency form one doubly degenerate level, which is how a
    # spectrum of a symmetric crystal shows a single line where there are two modes
    raman = make_raman(
        [np.eye(3), np.diag((1.0, -1.0, 0.0)), np.diag((0.0, 1.0, -1.0))],
        frequencies=(100.0, 300.0, 300.0),
    )
    rows = str(raman).splitlines()[3:]
    degeneracies = [int(row.split()[1]) for row in rows]
    assert degeneracies == [1, 2, 2]


def test_print(two_modes, format_):
    actual, _ = format_(two_modes)
    expected_text = """\
Raman activity at a laser energy of 0.00 eV
-------------------------------------------
mode  deg   omega (cm-1)   omega (meV)      activity   depolarization
   1    1         100.00         12.40       45.0000           0.0000
   2    1         500.00         61.99       21.0000           0.7500"""
    assert actual == {"text/plain": expected_text}


BOLTZMANN = 8.617333262e-5  # eV/K, the CODATA value


def test_intensity_at_zero_temperature_scales_the_activity(raman, Assert):
    # what a spectrometer sees is the activity times the fourth power of the frequency
    # of the scattered photon, divided by the frequency of the vibration
    laser = raman.read()["energies"][-1]
    activity = raman.activity(laser=laser)
    frequencies = activity["frequencies"]
    expected = activity["powder"] * (laser - frequencies) ** 4 / frequencies
    Assert.allclose(raman.intensity(laser=laser)["powder"], expected)


def test_intensity_applies_the_bose_factor(raman, Assert):
    # a warm crystal is already vibrating, so it scatters more than a cold one by the
    # Stokes factor n + 1
    laser = raman.read()["energies"][-1]
    cold = raman.intensity(laser=laser)
    hot = raman.intensity(laser=laser, temperature=300.0)
    ratio = hot["powder"] / cold["powder"]
    exponent = cold["frequencies"] / (BOLTZMANN * 300.0)
    Assert.allclose(ratio, 1 / (1 - np.exp(-exponent)))


def test_intensity_at_zero_temperature_has_no_bose_factor(raman, Assert):
    # the limit of n + 1 as the temperature goes to zero is one, not zero
    laser = raman.read()["energies"][-1]
    Assert.allclose(
        raman.intensity(laser=laser)["powder"],
        raman.intensity(laser=laser, temperature=1e-8)["powder"],
    )


def test_intensity_vanishes_for_modes_above_the_laser(raman):
    # the laser cannot excite a vibration that costs more energy than its photons
    laser = raman.read()["energies"][2]
    result = raman.intensity(laser=laser)
    assert np.any(result["frequencies"] > laser)
    assert np.all(result["powder"][result["frequencies"] >= laser] == 0)
    assert np.any(result["powder"] > 0)


def test_intensity_accepts_the_same_observables(raman):
    laser = raman.read()["energies"][-1]
    result = raman.intensity("parallel, perpendicular", laser=laser)
    assert sorted(result) == ["frequencies", "laser", "parallel", "perpendicular"]


def test_negative_temperature_raises_error(raman):
    with pytest.raises(exception.IncorrectUsage):
        raman.intensity(laser=raman.read()["energies"][-1], temperature=-1.0)


def test_to_graph_default_is_the_activity(raman):
    # plotting a thermally weighted spectrum by accident is the failure this prevents
    assert raman.to_graph().ylabel == "Raman activity (1/meV)"


def test_to_graph_with_temperature_plots_the_intensity(raman, Assert):
    laser = raman.read()["energies"][-1]
    shape = broadening.Gaussian(fwhm=1e-3)
    graph = raman.to_graph(laser=laser, temperature=300.0, shape=shape)
    assert graph.ylabel == "Raman intensity (1/meV)"
    total = np.sum(raman.intensity(laser=laser, temperature=300.0)["powder"])
    series = graph.series[0]
    Assert.allclose(np.trapezoid(series.y, series.x), total)


def test_to_graph_at_zero_temperature_still_plots_the_intensity(raman):
    # temperature=0 asks for the intensity without the thermal factor, which is not
    # the same curve as the bare activity
    laser = raman.read()["energies"][-1]
    shape = broadening.Gaussian(fwhm=1e-3)
    cold = raman.to_graph(laser=laser, temperature=0.0, shape=shape)
    activity = raman.to_graph(laser=laser, shape=shape)
    assert cold.ylabel != activity.ylabel
    assert np.max(cold.series[0].y) != np.max(activity.series[0].y)


def test_to_graph_with_temperature_needs_a_laser(raman):
    with pytest.raises(exception.IncorrectUsage):
        raman.to_graph(temperature=300.0)


def test_excitation_profile(raman, Assert):
    graph = raman.excitation_profile()
    frequencies = raman.read()["frequencies"]
    assert len(graph.series) == len(frequencies)
    assert graph.xlabel == "Laser energy (eV)"
    assert "activity" in graph.ylabel
    Assert.allclose(graph.series[0].x, raman.read()["energies"])


def test_excitation_profile_matches_the_activity_at_each_laser_energy(raman, Assert):
    energies = raman.read()["energies"]
    graph = raman.excitation_profile()
    for index in (0, 7, len(energies) - 1):
        expected = raman.activity(laser=energies[index])["powder"]
        actual = np.array([series.y[index] for series in graph.series])
        Assert.allclose(actual, expected)


def test_excitation_profile_labels_the_modes_by_their_wavenumber(raman):
    graph = raman.excitation_profile()
    wavenumbers = raman.read()["frequencies"] * convert.EV_TO_CM1
    assert graph.series[0].label == f"powder {wavenumbers[0]:.0f} cm-1"


def test_excitation_profile_selects_modes(raman):
    # the modes are numbered the way print labels them, counting from one
    graph = raman.excitation_profile(modes=[1, 3])
    wavenumbers = raman.read()["frequencies"] * convert.EV_TO_CM1
    labels = [series.label for series in graph.series]
    assert labels == [
        f"powder {wavenumbers[0]:.0f} cm-1",
        f"powder {wavenumbers[2]:.0f} cm-1",
    ]


def test_excitation_profile_with_temperature(raman, Assert):
    energies = raman.read()["energies"]
    graph = raman.excitation_profile(modes=[2], temperature=300.0)
    assert "intensity" in graph.ylabel
    index = len(energies) - 1
    expected = raman.intensity(laser=energies[index], temperature=300.0)["powder"][1]
    Assert.allclose(graph.series[0].y[index], expected)


def test_excitation_profile_with_several_observables(raman):
    graph = raman.excitation_profile("parallel, perpendicular", modes=[1])
    assert [series.label for series in graph.series] == [
        f"parallel {raman.read()['frequencies'][0] * convert.EV_TO_CM1:.0f} cm-1",
        f"perpendicular {raman.read()['frequencies'][0] * convert.EV_TO_CM1:.0f} cm-1",
    ]


def test_excitation_profile_rejects_a_mode_that_does_not_exist(raman):
    with pytest.raises(exception.IncorrectUsage):
        raman.excitation_profile(modes=[0])
    with pytest.raises(exception.IncorrectUsage):
        raman.excitation_profile(modes=[999])


def test_to_database(raman):
    handler = RamanHandler.from_data(raman.ref.raw)
    model = handler.to_database()
    assert isinstance(model, RamanModel)
    activity = raman.activity()
    assert model.number_modes == len(activity["frequencies"])
    assert model.frequency_max == float(np.max(activity["frequencies"]))
    strongest = int(np.argmax(activity["powder"]))
    assert model.strongest_frequency == float(activity["frequencies"][strongest])
    assert model.strongest_activity == float(activity["powder"][strongest])
    assert model.photon_energy_max == float(np.max(raman.read()["energies"]))


def test_to_database_dispatch(raman):
    assert set(raman._to_database()) == {"raman"}
