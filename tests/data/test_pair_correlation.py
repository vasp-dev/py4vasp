from py4vasp.data import PairCorrelation
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
