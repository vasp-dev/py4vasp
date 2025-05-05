# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import doctest
import pathlib
import re
import types

import h5py
import pytest

from py4vasp import _calculation, calculation, exception
from py4vasp._calculation import Calculation, DefaultCalculationFactory
from py4vasp._raw.definition import DEFAULT_FILE
from py4vasp._raw.write import write

DEFAULT_HASH = "631e66d7-a541-4b40-b25b-532544f8e720"
OTHER_PATH_1 = doctest.register_optionflag("OTHER_PATH_1")
OTHER_PATH_2 = doctest.register_optionflag("OTHER_PATH_2")


def create_symlinks(tmp_path_factory, tmp_dirname: str, elements: list):
    tmp_path = tmp_path_factory.mktemp(tmp_dirname)
    # create hdf5 file
    filename = tmp_path / DEFAULT_FILE
    with h5py.File(filename, "w") as h5f:
        for el in elements:
            write(h5f, el)
    # create symbolic links for paths used in doctest
    path_to = tmp_path / "path/to"
    path_to.mkdir(parents=True)
    pathlib.Path(path_to / "calculation").symlink_to(tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def setup_doctest(raw_data, tmp_path_factory, not_core):

    # DEFAULT CONFIG: tmp_path
    raw_dos = raw_data.dos("Sr2TiO4 with_projectors")
    raw_band = raw_data.band("multiple with_projectors")
    raw_structure = raw_data.structure("Sr2TiO4")
    raw_energy = raw_data.energy("relax", randomize=True)
    tmp_path = create_symlinks(
        tmp_path_factory, "data1", [raw_dos, raw_band, raw_structure, raw_energy]
    )
    #

    # BAND.to_quiver CONFIG:
    # other_path
    # raw_dos = raw_data.dos("Ba2PbO4 with_projectors")
    raw_band = raw_data.band("spin_texture with_projectors")
    # raw_structure = raw_data.structure("Ba2PbO4")
    band_to_quiver_other_path_1 = create_symlinks(
        tmp_path_factory, "data2", [raw_dos, raw_band, raw_structure, raw_energy]
    )
    #

    ### ADD MORE PATHS?
    # - Create a new symlink for the data.
    # - Add the function name to the returned dict.
    #
    # "path" is default; "other_path_1" and "other_path_2" can be defined, but don't have to.
    # NEVER delete DEFAULT_HASH's "path" entry.

    return {
        DEFAULT_HASH: {"path": tmp_path},
        "Band.to_quiver": {
            "path": tmp_path,
            "other_path_1": band_to_quiver_other_path_1,
        },
    }


def get_examples():
    finder = doctest.DocTestFinder()
    try:
        examples = (
            finder.find(_calculation)
            + finder.find(_calculation.dos)
            + finder.find(_calculation.band)
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


def _split_doctest_list(
    example: doctest.DocTest, ref_key: str, setup_doctest
) -> list[tuple[doctest.DocTest, str]]:
    _split_doctest_list: list[doctest.DocTest] = []
    default_path_doctests = [
        ex
        for ex in example.examples
        if (
            (not (OTHER_PATH_1 in ex.options) and not (OTHER_PATH_2 in ex.options))
            or (
                (OTHER_PATH_1 in ex.options)
                and not ("other_path_1" in setup_doctest[ref_key])
            )
            or (
                (OTHER_PATH_2 in ex.options)
                and not ("other_path_2" in setup_doctest[ref_key])
            )
        )
    ]
    other_path_1_doctests = [
        ex
        for ex in example.examples
        if ((OTHER_PATH_1 in ex.options) and ("other_path_1" in setup_doctest[ref_key]))
    ]
    other_path_2_doctests = [
        ex
        for ex in example.examples
        if ((OTHER_PATH_2 in ex.options) and ("other_path_2" in setup_doctest[ref_key]))
    ]
    return [
        (
            doctest.DocTest(
                examples=dt,
                globs=example.globs.copy(),
                name=example.name,
                filename=example.filename,
                lineno=example.lineno,
                docstring=example.docstring,
            ),
            p,
        )
        for (dt, p) in [
            (default_path_doctests, "path"),
            (other_path_1_doctests, "other_path_1"),
            (other_path_2_doctests, "other_path_2"),
        ]
        if dt
    ]


@pytest.mark.parametrize("example", get_examples(), ids=lambda example: example.name)
def test_example(example: doctest.DocTest, setup_doctest, monkeypatch):
    # Find all functions in examples that have a custom definition in setup_doctest
    path_special_keys: list = list(setup_doctest.keys())

    ref_key: str = DEFAULT_HASH
    for k in path_special_keys[1:]:
        if example.name.endswith(k):
            ref_key = k
            break

    # split list of examples according to requested path environment
    split_doctest_list = _split_doctest_list(example, ref_key, setup_doctest)

    # treat each case separately:
    for ex, sel_path_key in split_doctest_list:
        # set selected environment path for data for this doctest
        selected_path = setup_doctest.get(ref_key, setup_doctest[DEFAULT_HASH]).get(
            sel_path_key
        )
        if selected_path is None:
            selected_path = setup_doctest[DEFAULT_HASH]["path"]

        # Monkeypatch `__getattr__` to inject the correct path for the Calculation object
        def custom_getattr(self, attr):
            # When a specific path is needed, we pass it to Calculation.from_path()
            if attr == "load":
                return lambda: Calculation.from_path(selected_path)
            calc = Calculation.from_path(selected_path)
            return getattr(calc, attr)

        # Apply the patch to the factory object
        monkeypatch.setattr(DefaultCalculationFactory, "__getattr__", custom_getattr)

        # Ensure correct path setup before running doctests
        monkeypatch.chdir(setup_doctest[DEFAULT_HASH]["path"])

        # Apply doctest options (ELLIPSIS, NORMALIZE_WHITESPACE)
        optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        runner = doctest.DocTestRunner(optionflags=optionflags)

        # Ensure calculation is properly set in the globals
        ex.globs["calculation"] = calculation

        # Run the doctest
        result = runner.run(ex)

        # Assert that there are no failed tests
        assert (
            result.failed == 0
        ), f"assert {result.failed} == {0} @ data = {selected_path}"
