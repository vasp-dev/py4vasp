# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import doctest
import pathlib

import h5py
import pytest

import py4vasp
from py4vasp import _calculation, calculation, demo, exception
from py4vasp._raw.definition import DEFAULT_FILE
from py4vasp._raw.write import write

# ALTERNATIVE = doctest.register_optionflag("ALTERNATIVE")


# @pytest.fixture(scope="module")
# def setup_doctest(raw_data, tmp_path_factory, not_core):
#     tmp_path = tmp_path_factory.mktemp("default")

#     default_path = create_default_hdf5_file(raw_data, tmp_path)
#     spin_texture_path = create_spin_texture_hdf5_file(raw_data, tmp_path)
#     create_symbolic_links(tmp_path)
#     return {"path": default_path, "band.Band.to_quiver": spin_texture_path}


# def create_default_hdf5_file(raw_data, tmp_path):
#     filename = tmp_path / DEFAULT_FILE
#     raw_dos = raw_data.dos("Sr2TiO4 with_projectors")
#     raw_band = raw_data.band("multiple with_projectors")
#     raw_structure = raw_data.structure("Sr2TiO4")
#     raw_energy = raw_data.energy("relax", randomize=True)
#     with h5py.File(filename, "w") as h5f:
#         write(h5f, raw_dos)
#         write(h5f, raw_band)
#         write(h5f, raw_structure)
#         write(h5f, raw_energy)
#     return tmp_path


# def create_spin_texture_hdf5_file(raw_data, tmp_path):
#     spin_texture_path = tmp_path / "spin_texture"
#     spin_texture_path.mkdir()
#     filename = spin_texture_path / DEFAULT_FILE
#     raw_band = raw_data.band("spin_texture with_projectors")
#     with h5py.File(filename, "w") as h5f:
#         write(h5f, raw_band)
#     return spin_texture_path


# def create_symbolic_links(tmp_path):
#     path_to = tmp_path / "path/to"
#     path_to.mkdir(parents=True)
#     pathlib.Path(path_to / "calculation").symlink_to(tmp_path)


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
    print(example.name)
    optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    runner = doctest.DocTestRunner(optionflags=optionflags)
    example.globs["py4vasp"] = py4vasp
    example.globs["path"] = tmp_path
    result = runner.run(example)
    assert result.failed == 0


# @pytest.mark.parametrize("example", get_examples(), ids=lambda example: example.name)
# def test_example(example: doctest.DocTest, setup_doctest, monkeypatch):
#     example_path = setup_doctest.get(
#         example.name.removeprefix("py4vasp._calculation."), setup_doctest["path"]
#     )
#     monkeypatch.chdir(example_path)
#     optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
#     runner = doctest.DocTestRunner(optionflags=optionflags)
#     example.globs["calculation"] = calculation
#     result = runner.run(example)
#     assert result.failed == 0
