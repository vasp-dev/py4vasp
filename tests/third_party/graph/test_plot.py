# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._third_party.graph import Graph, Series, plot
from py4vasp._util import convert


def test_plot():
    x1, y1 = np.random.random((2, 50))
    series0 = Series(x1, y1)
    series1 = Series(x1, y1, "label1")
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
    weight = np.zeros(20)
    with pytest.raises(exception.IncorrectUsage):
        plot(x, y, weight=weight)


def test_many_colors(not_core):
    data = np.random.random((10, 2, 50))
    plots = (plot(x, y) for x, y in data)
    graph = sum(plots, start=Graph([]))
    figure = graph.to_plotly()
    assert len(figure.data) == 10
    colors = {series.line.color for series in figure.data}
    assert len(colors) > 4
    for color1, color2 in itertools.combinations(colors, 2):
        assert color_distance(color1, color2) > 30


def color_distance(color1, color2):
    lab1 = convert.to_lab(color1)
    lab2 = convert.to_lab(color2)
    return np.linalg.norm(lab1 - lab2)
