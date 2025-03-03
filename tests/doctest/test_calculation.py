# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import doctest
import pathlib

import h5py
import pytest

from py4vasp import _calculation, calculation
from py4vasp._raw.definition import DEFAULT_FILE
from py4vasp._raw.write import write


@pytest.fixture(scope="module")
def setup_doctest(raw_data, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("data")
    # create hdf5 file
    filename = tmp_path / DEFAULT_FILE
    raw_dos = raw_data.dos("Sr2TiO4 with_projectors")
    raw_structure = raw_data.structure("Sr2TiO4")
    raw_energy = raw_data.energy("relax", randomize=True)
    with h5py.File(filename, "w") as h5f:
        write(h5f, raw_dos)
        write(h5f, raw_structure)
        write(h5f, raw_energy)
    # create symbolic links for paths used in doctest
    path_to = tmp_path / "path/to"
    path_to.mkdir(parents=True)
    pathlib.Path(path_to / "calculation").symlink_to(tmp_path)
    #
    return {"path": tmp_path}


def get_examples():
    finder = doctest.DocTestFinder()
    examples = finder.find(_calculation) + finder.find(_calculation.dos)
    return [example for example in examples if interesting_example(example)]


def interesting_example(example):
    suffix = example.name.split(".")[-1]
    if len(example.examples) == 0:
        return False
    skipped_suffixes = (
        "band",
        "bandgap",
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
    optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    runner = doctest.DocTestRunner(optionflags=optionflags)
    example.globs["calculation"] = calculation
    result = runner.run(example)
    assert result.failed == 0
