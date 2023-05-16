# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._util import convert
from py4vasp.data import Energy


@pytest.fixture
def MD_energy(raw_data):
    raw_energy = raw_data.energy("MD")
    MD_energy = Energy.from_data(raw_energy)
    MD_energy.ref = types.SimpleNamespace()
    MD_energy.ref.number_steps = len(raw_energy.values)
    get_label = lambda x: convert.text_to_string(x).strip()
    MD_energy.ref.labels = [get_label(label) for label in raw_energy.labels]
    MD_energy.ref.values = raw_energy.values.T
    return MD_energy


@pytest.mark.parametrize(
    "selection, labels, subset",
    [
        (None, None, slice(None)),  # default selection = all
        ("temperature", ["temperature"], slice(3, 4)),
        ("TOTEN, EKIN", ["TOTEN", "EKIN"], slice(None, 2)),
    ],
)
@pytest.mark.parametrize("steps", [slice(None), slice(1, 3), 0, -1])
def test_read(selection, labels, subset, steps, MD_energy, Assert):
    kwargs = {"selection": selection} if selection else {}
    dict_ = MD_energy[steps].read(**kwargs) if steps != -1 else MD_energy.read(**kwargs)
    reference = MD_energy.ref
    labels = labels or reference.labels[subset]
    assert len(dict_) == len(labels)
    for label, expected in zip(labels, reference.values[subset]):
        Assert.allclose(dict_[label], expected[steps])


@pytest.mark.parametrize(
    "selection, subset",
    [
        (None, slice(0, 1)),  # default selection = TOTEN
        ("temperature", slice(3, 4)),
        ("ETOTAL, TEIN", [6, 3]),
    ],
)
@pytest.mark.parametrize("steps", [slice(None), slice(1, 3), 0, -1])
def test_plot(selection, subset, steps, MD_energy, Assert):
    kwargs = {"selection": selection} if selection else {}
    graph = MD_energy[steps].plot(**kwargs) if steps != -1 else MD_energy.plot(**kwargs)
    assert graph.xlabel == "Step"
    ylabel = "Temperature (K)" if selection == "temperature" else "Energy (eV)"
    assert graph.ylabel == ylabel
    y2label = "Temperature (K)" if selection == "ETOTAL, TEIN" else None
    assert graph.y2label == y2label
    xx = np.arange(MD_energy.ref.number_steps) + 1
    assert len(graph.series) == len(MD_energy.ref.values[subset])
    for series, yy in zip(graph.series, MD_energy.ref.values[subset]):
        Assert.allclose(series.x, xx[steps])
        Assert.allclose(series.y, yy[steps])


@pytest.mark.parametrize(
    "selection, subset",
    [
        (None, slice(0, 1)),  # default selection = TOTEN
        ("EPS", slice(5, 6)),  # TODO should be nose_kinetic
        ("EKIN_LAT, ES", [2, 4]),
    ],
)
@pytest.mark.parametrize("steps", [slice(None), slice(1, 3), 0, -1])
def test_to_numpy(selection, subset, steps, MD_energy, Assert):
    kwargs = {"selection": selection} if selection else {}
    if steps != -1:
        actual = MD_energy[steps].to_numpy(**kwargs)
    else:
        actual = MD_energy.to_numpy(**kwargs)
    expected_ndim = isinstance(steps, slice) + (selection == "EKIN_LAT, ES")
    assert actual.ndim == expected_ndim
    Assert.allclose(actual, MD_energy.ref.values[subset, steps])


def test_incorrect_label(MD_energy):
    with pytest.raises(exception.IncorrectUsage):
        MD_energy.read("not available")
    with pytest.raises(exception.IncorrectUsage):
        MD_energy.plot("not available")
    with pytest.raises(exception.IncorrectUsage):
        number_instead_of_string = 1
        MD_energy.plot(number_instead_of_string)


@patch("py4vasp._data.energy.Energy.to_graph")
def test_energy_to_plotly(mock_plot, MD_energy):
    fig = MD_energy.to_plotly("selection")
    mock_plot.assert_called_once_with("selection")
    graph = mock_plot.return_value
    graph.to_plotly.assert_called_once()
    assert fig == graph.to_plotly.return_value


def test_to_image(MD_energy):
    check_to_image(MD_energy, None, "energy.png")
    custom_filename = "custom.jpg"
    check_to_image(MD_energy, custom_filename, custom_filename)


def check_to_image(MD_energy, filename_argument, expected_filename):
    with patch("py4vasp._data.energy.Energy.to_plotly") as plot:
        MD_energy.to_image("args", filename=filename_argument, key="word")
        plot.assert_called_once_with("args", key="word")
        fig = plot.return_value
        fig.write_image.assert_called_once_with(MD_energy._path / expected_filename)


def test_labels(MD_energy):
    total_energy = MD_energy.ref.labels[0]
    kinetic_energy = MD_energy.ref.labels[1]
    temperature = MD_energy.ref.labels[3]
    assert MD_energy.labels() == MD_energy.ref.labels
    assert MD_energy.labels("temperature") == [temperature]
    assert MD_energy.labels("TOTEN, EKIN") == [total_energy, kinetic_energy]


@pytest.mark.parametrize(
    "steps, step_label",
    [
        (-1, "final step"),
        (0, "step 1"),
        (slice(None), "step 4 of range 1:4"),
        (slice(1, 3), "step 3 of range 2:3"),
    ],
)
def test_print(steps, step_label, MD_energy, format_):
    actual, _ = format_(MD_energy[steps]) if steps != -1 else format_(MD_energy)
    if isinstance(steps, int):
        last_step = steps
    else:
        last_step = (steps.stop or MD_energy.ref.number_steps) - 1
    energies = MD_energy.ref.values[:, last_step]
    lines = [f"Energies at {step_label}:"]
    lines += [
        f"   {ll:23.23}={ee:17.6f}" for ll, ee in zip(MD_energy.labels(), energies)
    ]
    assert actual == {"text/plain": "\n".join(lines)}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.energy("MD")
    check_factory_methods(Energy, data)
