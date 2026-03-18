# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import doctest
import pathlib
from unittest.mock import patch

import numpy as np
import pytest

import py4vasp
from py4vasp import _calculation, demo, exception


def test_creating_default_calculation(tmp_path):
    demo.calculation(tmp_path / "specific_example")


finder = doctest.DocTestFinder()


def find_examples(obj):
    try:
        return finder.find(obj)
    except exception.ModuleNotInstalled:
        return []


def get_calculation_examples():
    examples = (
        find_examples(_calculation)
        + find_examples(_calculation.band)
        + find_examples(_calculation.dos)
        + find_examples(_calculation.force)
        + find_examples(_calculation.local_moment)
        + find_examples(_calculation.structure)
    )
    return [example for example in examples if interesting_example(example)]


def interesting_example(example):
    suffix = example.name.split(".")[-1]
    if len(example.examples) == 0:
        return False
    skipped_suffixes = (
        "bandgap",
        "energy",
        "pair_correlation",
    )
    return suffix not in skipped_suffixes


@pytest.mark.parametrize(
    "example", get_calculation_examples(), ids=lambda example: example.name
)
def test_calculation(example: doctest.DocTest, tmp_path: pathlib.Path):
    optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    runner = doctest.DocTestRunner(optionflags=optionflags)
    example.globs["py4vasp"] = py4vasp
    example.globs["path"] = tmp_path / example.name.replace(".", "_")
    result = runner.run(example)
    assert result.failed == 0
    assert result.attempted > 0


def get_graph_examples():
    return (
        find_examples(py4vasp.plot)
        + find_examples(py4vasp.graph.Contour)
        + find_examples(py4vasp.graph.Graph)
        + find_examples(py4vasp.graph.Series)
    )


@pytest.mark.parametrize(
    "example", get_graph_examples(), ids=lambda example: example.name
)
def test_graph_functions(example: doctest.DocTest, tmp_path: pathlib.Path, not_core):
    optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    runner = doctest.DocTestRunner(optionflags=optionflags)
    example.globs["np"] = np
    example.globs["py4vasp"] = py4vasp
    example.globs["path"] = tmp_path
    with patch("plotly.graph_objs.Figure.show", return_value=None):
        result = runner.run(example)
    assert result.failed == 0
    assert result.attempted > 0
