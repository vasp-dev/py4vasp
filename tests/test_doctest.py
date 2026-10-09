# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import ast
import doctest
import importlib
import pathlib
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

import py4vasp
from py4vasp import _calculation, broadening, demo
from py4vasp._calculation import (  # noqa: F401 — imports submodules as _calculation attributes
    band,
    bandgap,
    born_effective_charge,
    dielectric_function,
    dielectric_tensor,
    dos,
    elastic_modulus,
    energy,
    force,
    force_constant,
    kpoint,
    local_moment,
    mass,
    neighbor_list,
    optics,
    pair_correlation,
    partial_density,
    phonon_band,
    phonon_dos,
    phonon_mode,
    projector,
    raman,
    reaction_path,
    stress,
    structure,
    symmetry,
    system,
    velocity,
    workfunction,
)
from py4vasp._third_party import numeric as _numeric
from py4vasp._util import color as _util_color
from py4vasp._util import import_

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
        + find_examples(_calculation.bandgap)
        + find_examples(_calculation.born_effective_charge)
        + find_examples(_calculation.dielectric_function)
        + find_examples(_calculation.dielectric_tensor)
        + find_examples(_calculation.dos)
        + find_examples(_calculation.elastic_modulus)
        + find_examples(_calculation.energy)
        + find_examples(_calculation.force)
        + find_examples(_calculation.force_constant)
        + find_examples(_calculation.kpoint)
        + find_examples(_calculation.local_moment)
        + find_examples(_calculation.mass)
        + find_examples(_calculation.neighbor_list)
        + find_examples(_calculation.optics)
        + find_examples(_calculation.pair_correlation)
        + find_examples(_calculation.partial_density)
        + find_examples(_calculation.phonon_band)
        + find_examples(_calculation.phonon_dos)
        + find_examples(_calculation.phonon_mode)
        + find_examples(_calculation.projector)
        + find_examples(_calculation.raman)
        + find_examples(_calculation.reaction_path)
        + find_examples(_calculation.stress)
        + find_examples(_calculation.structure)
        + find_examples(_calculation.symmetry)
        + find_examples(_calculation.system)
        + find_examples(_calculation.velocity)
        + find_examples(_calculation.workfunction)
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
    "py4vasp._calculation.partial_density.PartialDensity.to_stm": "scipy",
    "py4vasp._calculation.symmetry.Symmetry.space_group": "spglib",
    "py4vasp._calculation.symmetry.Symmetry.point_group_schoenflies": "spglib",
    "py4vasp._calculation.symmetry.Symmetry.bravais_lattice": "spglib",
    "py4vasp._calculation.symmetry.Symmetry.pearson_symbol": "spglib",
    "py4vasp._calculation.structure.Structure.wyckoff_positions": "spglib",
    "py4vasp._calculation.structure.Structure.standardized_cell": "spglib",
    "py4vasp._calculation.structure.Structure.prototype": "spglib",
    "py4vasp._calculation.structure.Structure.symmetrize": "spglib",
    "py4vasp._calculation.structure.Structure.generate_kpath": "seekpath",
    "py4vasp._calculation.structure.Structure.generate_kmesh": "spglib",
    "py4vasp._calculation.structure.Structure.conventional_lattice_vectors": "spglib",
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
    if len(example.examples) == 0:
        return False
    # Every module with examples is collected now, so nothing has to be filtered by
    # name. Add a suffix here if a docstring gains an example that cannot be executed.
    return True


def _run_example(example, tmp_path, monkeypatch):
    # examples may change the directory or create temporary ones; both are confined to
    # this test and undone afterwards
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    runner = doctest.DocTestRunner(optionflags=optionflags)
    example.globs["py4vasp"] = py4vasp
    result = runner.run(example)
    assert result.failed == 0
    assert result.attempted > 0


@pytest.mark.parametrize(
    "example", get_calculation_examples(), ids=lambda example: example.name
)
def test_calculation(example: doctest.DocTest, tmp_path: pathlib.Path, monkeypatch):
    _run_example(example, tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "example", get_full_calculation_examples(), ids=lambda example: example.name
)
def test_calculation_full(
    example: doctest.DocTest, tmp_path: pathlib.Path, monkeypatch
):
    pytest.importorskip(_FULL_INSTALL_EXAMPLES[example.name])
    _run_example(example, tmp_path, monkeypatch)


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


def get_demo_examples():
    return [example for example in find_examples(demo) if interesting_example(example)]


@pytest.mark.parametrize(
    "example", get_demo_examples(), ids=lambda example: example.name
)
def test_demo(example: doctest.DocTest, tmp_path: pathlib.Path, monkeypatch):
    _run_example(example, tmp_path, monkeypatch)


def get_broadening_examples():
    # the public module only re-exports; the classes and the function are defined in
    # _third_party.numeric, so both have to be searched to reach every example
    examples = find_examples(broadening) + find_examples(_numeric)
    return [example for example in examples if interesting_example(example)]


@pytest.mark.parametrize(
    "example", get_broadening_examples(), ids=lambda example: example.name
)
def test_broadening(example: doctest.DocTest):
    # deliberately no importorskip: broadening is pure numpy and must run on the core
    # installation, unlike the interpolation routines that share its module
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
def test_graph_functions(example: doctest.DocTest, tmp_path: pathlib.Path, monkeypatch):
    pytest.importorskip("plotly")
    example.globs["np"] = np
    with patch("plotly.graph_objs.Figure.show", return_value=None):
        _run_example(example, tmp_path, monkeypatch)


def _example_from_source(source, filename):
    parser = doctest.DocTestParser()
    return parser.get_doctest(source, {}, "example", filename, 0)


def _reads_undefined_path(example):
    "Does the example use a variable `path` before it assigns one?"
    defined = False
    for line in example.examples:
        tree = ast.parse(line.source)
        names = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id == "path"
        ]
        if not defined and any(isinstance(n.ctx, ast.Load) for n in names):
            return True
        defined = defined or any(isinstance(n.ctx, ast.Store) for n in names)
    return False


@pytest.mark.parametrize(
    "source, expected",
    (
        (">>> path\n", True),
        (">>> calculation.path()\n", False),
        (">>> f(path='folder')\n", False),
        (">>> path = 'folder'\n>>> print(path)\n", False),
    ),
)
def test_reads_undefined_path(source, expected):
    assert _reads_undefined_path(_example_from_source(source, "x.py")) == expected


def _every_module_with_examples():
    # also the modules whose examples are not executed yet, because a user copies those
    # just the same
    directory = pathlib.Path(_calculation.__file__).parent
    yield _calculation
    for path in sorted(directory.glob("[a-z]*.py")):
        yield importlib.import_module(f"py4vasp._calculation.{path.stem}")
    yield from (demo, py4vasp._third_party.graph.graph, py4vasp._third_party.view.view)


def test_no_example_reads_an_undefined_path():
    # The examples used to rely on a `path` injected by this test, so they failed with
    # a NameError when a user copied them. Nothing is injected anymore, but the
    # examples that are not executed would not notice a regression.
    offenders = sorted(
        example.name
        for module in _every_module_with_examples()
        for example in find_examples(module)
        if _reads_undefined_path(example)
    )
    assert not offenders, f"{offenders} use a variable path they never define"


def test_examples_found_despite_missing_optional_module(monkeypatch):
    # structure.py declares the optional mdtraj, which is deliberately kept out of the
    # `all` extra, so this is the state of every CI job. A lazy proxy for an absent
    # package must not hide the examples of the whole module it is declared in.
    monkeypatch.setattr(
        _calculation.structure, "mdtraj", import_.optional("_not_installed_")
    )
    names = [example.name for example in find_examples(_calculation.structure)]
    assert "py4vasp._calculation.structure.Structure.read" in names


# Modules whose examples call demo.calculation but still cannot be collected: their
# method-level examples build a calculation with py4vasp.Calculation.from_path(".")
# instead, which has no data to read. They need those examples rewritten onto the demo
# before they can join the list above; until then they are knowingly excluded here so
# that a module going uncollected by accident is still reported.
_EXAMPLES_NOT_RUNNABLE_YET = (
    "current_density",
    "density",
    "exciton_density",
    "nics",
    "potential",
)


def _modules_building_their_own_data():
    """Names of the _calculation modules whose examples create the data they need."""
    directory = pathlib.Path(_calculation.__file__).parent
    for path in sorted(directory.glob("[a-z]*.py")):
        module = importlib.import_module(f"py4vasp._calculation.{path.stem}")
        if any(_builds_a_calculation(example) for example in find_examples(module)):
            yield path.stem


def _builds_a_calculation(example):
    return any("demo.calculation(" in line.source for line in example.examples)


def test_every_self_contained_example_is_collected():
    # _all_calculation_examples enumerates the modules by hand, so a module is silently
    # dropped when nobody remembers to add it. Every example of every module dropped
    # that way stops running, which is how the Structure examples once vanished.
    collected = {example.name.split(".")[2] for example in _all_calculation_examples()}
    candidates = set(_modules_building_their_own_data())
    assert candidates, "no module builds its own data, so the search above is broken"
    forgotten = sorted(candidates - collected - set(_EXAMPLES_NOT_RUNNABLE_YET))
    assert not forgotten, f"examples of {forgotten} are never executed"
    stale = sorted(set(_EXAMPLES_NOT_RUNNABLE_YET) & collected)
    assert not stale, f"{stale} are collected now, so drop them from the exclusion"
