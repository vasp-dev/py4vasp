from py4vasp.data import PairCorrelation
import py4vasp.exceptions as exception
import pytest


@pytest.fixture
def pair_correlation(raw_data):
    raw_pair_correlation = raw_data.pair_correlation("Sr2TiO4")
    pair_correlation = PairCorrelation(raw_pair_correlation)
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


def test_plot_nonexisting_label(pair_correlation):
    with pytest.raises(exception.IncorrectUsage):
        pair_correlation.plot("label does exist")
