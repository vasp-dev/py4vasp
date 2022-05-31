from py4vasp.data import PairCorrelation
import py4vasp.exceptions as exception
import pytest
from unittest.mock import patch


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
    assert fig.series[0].name == "total"
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
    for series, (name, index) in zip(fig.series, expected.items()):
        assert series.name == name
        Assert.allclose(series.x, pair_correlation.ref.distances)
        Assert.allclose(series.y, pair_correlation.ref.function[steps, index])


def test_labels(pair_correlation):
    assert pair_correlation.labels() == pair_correlation.ref.labels


def test_plot_nonexisting_label(pair_correlation):
    with pytest.raises(exception.IncorrectUsage):
        pair_correlation.plot("label does exist")


@patch("py4vasp.data.pair_correlation.PairCorrelation.plot")
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
    with patch("py4vasp.data.pair_correlation.PairCorrelation.to_plotly") as plot:
        pair_correlation.to_image("args", filename=filename_argument, key="word")
        plot.assert_called_once_with("args", key="word")
        fig = plot.return_value
        expected_path = pair_correlation.path / expected_filename
        fig.write_image.assert_called_once_with(expected_path)


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.pair_correlation("Sr2TiO4")
    check_factory_methods(PairCorrelation, data)
