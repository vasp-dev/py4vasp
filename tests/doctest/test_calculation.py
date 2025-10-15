# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import doctest
import pathlib

import pytest

import py4vasp
from py4vasp import _calculation, demo, exception


def test_creating_default_calculation(tmp_path):
    demo.calculation(tmp_path)


def get_calculation_examples():
    finder = doctest.DocTestFinder()
    try:
        examples = (
            finder.find(_calculation)
            # + finder.find(_calculation.dos)
            # + finder.find(_calculation.band)
        )
    except exception.ModuleNotInstalled:
        return []
    return [example for example in examples if interesting_example(example)]


def interesting_example(example):
    suffix = example.name.split(".")[-1]
    if len(example.examples) == 0:
        return False
    skipped_suffixes = (
        "bandgap",
        "energy",
        "force",
        "local_moment",
        "pair_correlation",
        "stress",
        "structure",
        "velocity",
    )
    return suffix not in skipped_suffixes


@pytest.mark.parametrize(
    "example", get_calculation_examples(), ids=lambda example: example.name
)
def test_calculation(example: doctest.DocTest, tmp_path: pathlib.Path):
    optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    runner = doctest.DocTestRunner(optionflags=optionflags)
    example.globs["py4vasp"] = py4vasp
    example.globs["path"] = tmp_path
    result = runner.run(example)
    assert result.failed == 0
    assert result.attempted > 0
