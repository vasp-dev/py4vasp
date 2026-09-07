# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Run the examples in vasp.backend the same way tests/test_doctest.py runs py4vasp's."""

import doctest
import pathlib

import pytest
from vasp import backend

import py4vasp

finder = doctest.DocTestFinder()


def get_examples():
    return [example for example in finder.find(backend) if example.examples]


def test_every_wrapper_has_an_example():
    """Every callable defined here needs an example.

    Only the two dataclass re-exports are exempt, and only because they are data
    containers whose fields py4vasp documents; every function is wrapped in this module
    precisely so that it carries its own documentation and example.
    """
    documented = {example.name.split(".")[-1] for example in get_examples()}
    wrappers = {
        name
        for name in backend.__all__
        if getattr(getattr(backend, name), "__module__", None) == backend.__name__
    }
    assert wrappers
    assert wrappers <= documented


@pytest.mark.parametrize("example", get_examples(), ids=lambda example: example.name)
def test_example(example: doctest.DocTest, tmp_path: pathlib.Path):
    optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    runner = doctest.DocTestRunner(optionflags=optionflags)
    example.globs["py4vasp"] = py4vasp
    example.globs["path"] = tmp_path / example.name.replace(".", "_")
    result = runner.run(example)
    assert result.failed == 0
    assert result.attempted > 0
