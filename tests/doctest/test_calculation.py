# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import doctest

import h5py
import pytest

from py4vasp import _calculation, calculation
from py4vasp._raw.definition import DEFAULT_FILE
from py4vasp._raw.write import write


@pytest.fixture(scope="module")
def setup_doctest(raw_data, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("data")
    filename = tmp_path / DEFAULT_FILE
    raw_dos = raw_data.dos("Sr2TiO4 with_projectors")
    with h5py.File(filename, "w") as h5f:
        write(h5f, raw_dos)
    return {"path": tmp_path}


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
def test_example(example, setup_doctest, monkeypatch):
    monkeypatch.chdir(setup_doctest["path"])
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    result = runner.run(example)
    assert result.failed == 0
