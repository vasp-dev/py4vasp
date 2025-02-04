# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp import exception
from py4vasp._third_party.graph import Graph, Series, plot


def test_plot():
    x1, x2, y1, y2 = np.random.random((4, 50))
    series0 = Series(x1, y1)
    series1 = Series(x1, y1, "label1")
    series2 = Series(x2, y2, "label2")
    assert plot(x1, y1) == Graph(series0)
    assert plot(x1, y1, "label1") == Graph(series1)
    assert plot(x1, y1, label="label1") == Graph(series1)
    assert plot(x1, y1, xlabel="xaxis") == Graph(series0, xlabel="xaxis")
    assert plot(x1, y=y1) == Graph(series0)
    assert plot(x=x1, y=y1) == Graph(series0)


def test_plot_small_dataset():
    for length in range(10):
        x = np.linspace(0, 1, length)
        y = x**2
        series = Series(x, y)
        assert plot(x, y) == Graph(series)


def test_plot_inconsistent_length():
    x = np.zeros(10)
    y = np.zeros(20)
    with pytest.raises(exception.IncorrectUsage):
        plot(x, y)


def test_fatband_inconsistent_length():
    x = np.zeros(10)
    y = np.zeros((5, 10))
    width = np.zeros(20)
    with pytest.raises(exception.IncorrectUsage):
        plot(x, y, width=width)
