# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import doctest

import pytest

from py4vasp import _calculation


def get_examples():
    finder = doctest.DocTestFinder()
    examples = finder.find(_calculation)
    return [example for example in examples if interesting_example(example)]


def interesting_example(example):
    suffix = example.name.split(".")[-1]
    if len(example.examples) == 0:
        return False
    skipped_suffixes = (
        "Calculation",
        "DefaultCalculationFactory",
        "band",
        "bandgap",
        "dos",
        "energy",
        "force",
        "magnetism",
        "pair_correlation",
        "stress",
        "structure",
        "velocity",
    )
    return suffix not in skipped_suffixes


@pytest.mark.parametrize("example", get_examples(), ids=lambda example: example.name)
def test_example(example):
    runner = doctest.DocTestRunner(verbose=False)
    result = runner.run(example)
    assert result.failed == 0
