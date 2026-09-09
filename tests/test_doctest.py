# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import doctest
import pathlib
from unittest.mock import patch

import numpy as np
import pytest

import py4vasp
from py4vasp import _calculation, demo
from py4vasp._calculation import (  # noqa: F401 — imports submodules as _calculation attributes
    band,
    dos,
    force,
    kpoint,
    local_moment,
    neighbor_list,
    optics,
    phonon_band,
    projector,
    structure,
    symmetry,
    system,
)
from py4vasp._util import color as _util_color
from py4vasp._util import import_


def test_creating_default_calculation(tmp_path):
    demo.calculation(tmp_path / "specific_example")


def test_creating_perovskite_calculation(tmp_path):
    # the "perovskite" selection pairs a structure with its symmetry so the
    # symmetry-derived structure examples have consistent data
    calculation = demo.calculation(tmp_path / "perovskite_example", "perovskite")
    assert calculation.structure.number_atoms() == 5


finder = doctest.DocTestFinder()


def find_examples(obj):
    # Deliberately no try/except here: swallowing ModuleNotInstalled used to drop every
    # example of a module as soon as one optional dependency was absent, which silently
    # hid all Structure examples from every CI job. An example that needs an extra
    # belongs in _FULL_INSTALL_EXAMPLES below, where it is skipped individually.
    return finder.find(obj)


def _all_calculation_examples():
    examples = (
        find_examples(_calculation)
        + find_examples(_calculation.band)
        + find_examples(_calculation.dos)
        + find_examples(_calculation.force)
        + find_examples(_calculation.kpoint)
        + find_examples(_calculation.local_moment)
        + find_examples(_calculation.neighbor_list)
        + find_examples(_calculation.optics)
        + find_examples(_calculation.phonon_band)
        + find_examples(_calculation.projector)
        + find_examples(_calculation.structure)
        + find_examples(_calculation.symmetry)
        + find_examples(_calculation.system)
    )
    return [example for example in examples if interesting_example(example)]


# Examples that rely on an optional package (e.g. scipy for the color pipeline or spglib
# for the space-group analysis) which is not part of the py4vasp-core installation. Each is
# mapped to the package it needs; they run in test_calculation_full which skips via
# importorskip when that package is missing. Everything else runs in test_calculation.
_FULL_INSTALL_EXAMPLES = {
    "py4vasp._calculation.band.Band.to_frame": "pandas",
    "py4vasp._calculation.dos.Dos.to_frame": "pandas",
    "py4vasp._calculation.neighbor_list.NeighborList.read": "scipy",
    "py4vasp._calculation.neighbor_list.NeighborList.to_string": "scipy",
    "py4vasp._calculation.optics.Optics.color": "scipy",
    "py4vasp._calculation.symmetry.Symmetry.space_group": "spglib",
    "py4vasp._calculation.symmetry.Symmetry.point_group_schoenflies": "spglib",
    "py4vasp._calculation.symmetry.Symmetry.bravais_lattice": "spglib",
    "py4vasp._calculation.symmetry.Symmetry.pearson_symbol": "spglib",
    "py4vasp._calculation.structure.Structure.wyckoff_positions": "spglib",
    "py4vasp._calculation.structure.Structure.standardized_cell": "spglib",
    "py4vasp._calculation.structure.Structure.prototype": "spglib",
    "py4vasp._calculation.structure.Structure.symmetrize": "spglib",
    "py4vasp._calculation.structure.Structure.to_ase": "ase",
    "py4vasp._calculation.structure.Structure.to_lammps": "ase",
    "py4vasp._calculation.structure.Structure.to_mdtraj": "mdtraj",
}


def _requires_full_install(example):
    return example.name in _FULL_INSTALL_EXAMPLES


def get_calculation_examples():
    return [e for e in _all_calculation_examples() if not _requires_full_install(e)]


def get_full_calculation_examples():
    return [e for e in _all_calculation_examples() if _requires_full_install(e)]


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


def _run_calculation_example(example, tmp_path):
    optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    runner = doctest.DocTestRunner(optionflags=optionflags)
    example.globs["py4vasp"] = py4vasp
    example.globs["path"] = tmp_path / example.name.replace(".", "_")
    result = runner.run(example)
    assert result.failed == 0
    assert result.attempted > 0


@pytest.mark.parametrize(
    "example", get_calculation_examples(), ids=lambda example: example.name
)
def test_calculation(example: doctest.DocTest, tmp_path: pathlib.Path):
    _run_calculation_example(example, tmp_path)


@pytest.mark.parametrize(
    "example", get_full_calculation_examples(), ids=lambda example: example.name
)
def test_calculation_full(example: doctest.DocTest, tmp_path: pathlib.Path):
    pytest.importorskip(_FULL_INSTALL_EXAMPLES[example.name])
    _run_calculation_example(example, tmp_path)


def get_util_examples():
    examples = find_examples(_util_color)
    return [example for example in examples if interesting_example(example)]


@pytest.mark.parametrize(
    "example", get_util_examples(), ids=lambda example: example.name
)
def test_util(example: doctest.DocTest):
    optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    runner = doctest.DocTestRunner(optionflags=optionflags)
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
def test_graph_functions(example: doctest.DocTest, tmp_path: pathlib.Path):
    pytest.importorskip("plotly")
    optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    runner = doctest.DocTestRunner(optionflags=optionflags)
    example.globs["np"] = np
    example.globs["py4vasp"] = py4vasp
    example.globs["path"] = tmp_path
    with patch("plotly.graph_objs.Figure.show", return_value=None):
        result = runner.run(example)
    assert result.failed == 0
    assert result.attempted > 0


def test_examples_found_despite_missing_optional_module(monkeypatch):
    # structure.py declares the optional mdtraj, which is deliberately kept out of the
    # `all` extra, so this is the state of every CI job. A lazy proxy for an absent
    # package must not hide the examples of the whole module it is declared in.
    monkeypatch.setattr(
        _calculation.structure, "mdtraj", import_.optional("_not_installed_")
    )
    names = [example.name for example in find_examples(_calculation.structure)]
    assert "py4vasp._calculation.structure.Structure.read" in names
