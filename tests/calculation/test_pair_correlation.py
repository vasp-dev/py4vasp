# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception, raw
from py4vasp._calculation.pair_correlation import (
    PairCorrelation,
    PairCorrelationHandler,
)
from py4vasp._raw.data_db import PairCorrelation_DB


@pytest.fixture
def pair_correlation(raw_data):
    raw_pair_correlation = raw_data.pair_correlation("Sr2TiO4")
    pair_correlation = PairCorrelation.from_data(raw_pair_correlation)
    pair_correlation.ref = raw_pair_correlation
    return pair_correlation


def test_read_default(pair_correlation, Assert):
    for steps in (slice(None), slice(1, 3), 0):
        actual = pair_correlation[steps].read()
        check_read_default(pair_correlation, actual, steps, Assert)
    check_read_default(pair_correlation, pair_correlation.read(), -1, Assert)


def check_read_default(pair_correlation, dict_, steps, Assert):
    assert len(dict_) == len(pair_correlation.ref.labels) + 1
    Assert.allclose(dict_["distances"], pair_correlation.ref.distances)
    for i, label in enumerate(pair_correlation.ref.labels):
        Assert.allclose(dict_[label], pair_correlation.ref.function[steps, i])


def test_plot_default(pair_correlation, Assert):
    for steps in (0, slice(None), slice(1, 3)):
        actual = pair_correlation[steps].plot()
        check_plot_default(pair_correlation, actual, steps, Assert)
    check_plot_default(pair_correlation, pair_correlation.plot(), -1, Assert)


def check_plot_default(pair_correlation, fig, steps, Assert):
    assert fig.xlabel == "Distance (Å)"
    assert fig.ylabel == "Pair correlation"
    assert fig.series[0].label == "total"
    Assert.allclose(fig.series[0].x, pair_correlation.ref.distances)
    Assert.allclose(fig.series[0].y, pair_correlation.ref.function[steps, 0])


def test_plot_selection(pair_correlation, Assert):
    selection = "Sr~Ti O~Ti"
    for steps in (0, slice(None), slice(1, 3)):
        actual = pair_correlation[steps].plot(selection)
        check_plot_selection(pair_correlation, actual, steps, Assert)
    check_plot_selection(pair_correlation, pair_correlation.plot(selection), -1, Assert)


def check_plot_selection(pair_correlation, fig, steps, Assert):
    assert fig.xlabel == "Distance (Å)"
    assert fig.ylabel == "Pair correlation"
    expected = {"Sr~Ti": 2, "Ti~O": 5}  # note the reordering of the label
    for series, (label, index) in zip(fig.series, expected.items()):
        assert series.label == label
        Assert.allclose(series.x, pair_correlation.ref.distances)
        Assert.allclose(series.y, pair_correlation.ref.function[steps, index])


def test_labels(pair_correlation):
    assert pair_correlation.labels() == pair_correlation.ref.labels


def test_plot_nonexisting_label(pair_correlation):
    with pytest.raises(exception.IncorrectUsage):
        pair_correlation.plot("label does exist")


@patch.object(PairCorrelation, "to_graph")
def test_pair_correlation_to_plotly(mock_plot, pair_correlation):
    fig = pair_correlation.to_plotly("selection")
    mock_plot.assert_called_once_with("selection")
    graph = mock_plot.return_value
    graph.to_plotly.assert_called_once()
    assert fig == graph.to_plotly.return_value


def test_to_image(pair_correlation):
    check_to_image(pair_correlation, None, "pair_correlation.png")
    custom_filename = "custom.jpg"
    check_to_image(pair_correlation, custom_filename, custom_filename)


def check_to_image(pair_correlation, filename_argument, expected_filename):
    with patch.object(PairCorrelation, "to_plotly") as plot:
        pair_correlation.to_image("args", filename=filename_argument, key="word")
        plot.assert_called_once_with("args", key="word")
        fig = plot.return_value
        expected_path = pair_correlation.path / expected_filename
        fig.write_image.assert_called_once_with(expected_path)


def test_to_database(pair_correlation, raw_data):
    raw_pair_correlation = raw_data.pair_correlation("Sr2TiO4")
    handler = PairCorrelationHandler.from_data(raw_pair_correlation)
    db_data: PairCorrelation_DB = handler.to_database()
    assert isinstance(db_data, PairCorrelation_DB)
    assert db_data.distance_min == float(pair_correlation.ref.distances[0])
    assert db_data.distance_max == float(pair_correlation.ref.distances[-1])


def _pair_correlation_from_total(distances, total):
    """Build a raw pair correlation whose only curve is the given total g(r)."""
    function = np.asarray(total)[np.newaxis, np.newaxis, :]  # (steps, labels, points)
    return raw.PairCorrelation(
        distances=np.asarray(distances, dtype=float),
        function=function,
        labels=("total",),
    )


def test_to_database_first_peak():
    """The first peak is the first local maximum of the total g(r) above the
    threshold; a small sub-threshold bump before it must be ignored."""
    distances = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    # bump at index 1 (0.5) is below the threshold; the first real peak is at index 4
    total = [0.0, 0.5, 0.3, 2.5, 4.0, 1.5, 1.2, 1.0]
    raw_pcf = _pair_correlation_from_total(distances, total)
    db_data = PairCorrelationHandler.from_data(raw_pcf).to_database()
    assert isinstance(db_data, PairCorrelation_DB)
    assert db_data.first_peak_position == 4.0
    assert db_data.first_peak_height == 4.0


def test_to_database_no_first_peak():
    """A monotonic curve staying below the threshold has no detectable peak."""
    distances = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    total = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    raw_pcf = _pair_correlation_from_total(distances, total)
    db_data = PairCorrelationHandler.from_data(raw_pcf).to_database()
    assert db_data.first_peak_position is None
    assert db_data.first_peak_height is None


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.pair_correlation("Sr2TiO4")
    check_factory_methods(PairCorrelation, data)
