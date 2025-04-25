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

PATH_SPIN_TEXTURE = doctest.register_optionflag("PATH_SPIN_TEXTURE")


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

    # SPIN TEXTURE CONFIG: spin_texture_path
    raw_band = raw_data.band("spin_texture with_projectors")
    spin_texture_path = create_symlinks(
        tmp_path_factory, "data2", [raw_dos, raw_band, raw_structure, raw_energy]
    )
    #
    return {"path": tmp_path, "path_spin_texture": spin_texture_path}


class CustomDocTestParser(doctest.DocTestParser):
    def _find_options(self, source, name, lineno):
        """
        Override _find_options to handle custom flags like +PATH_SPIN_TEXTURE
        """
        options = {}
        # (note: with the current regexp, this will match at most once:)
        for m in self._OPTION_DIRECTIVE_RE.finditer(source):
            option_strings = m.group(1).replace(",", " ").split()
            for option in option_strings:
                # raise UserWarning(option)
                if (
                    option[0] not in "+-"
                    or option[1:] not in doctest.OPTIONFLAGS_BY_NAME
                ):
                    raise ValueError(
                        "line %r of the doctest for %s "
                        "has an invalid option: %r" % (lineno + 1, name, option)
                    )
                flag = doctest.OPTIONFLAGS_BY_NAME[option[1:]]
                options[flag] = option[0] == "+"
        if options and self._IS_BLANK_OR_COMMENT(source):
            raise ValueError(
                "line %r of the doctest for %s has an option "
                "directive on a line with no example: %r" % (lineno, name, source)
            )
        # if (options): raise UserWarning(options)
        return options


def get_examples():
    finder = doctest.DocTestFinder(parser=CustomDocTestParser())
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
        "magnetism",
        "pair_correlation",
        "stress",
        "structure",
        "velocity",
    )
    return suffix not in skipped_suffixes


@pytest.mark.parametrize("examples", get_examples(), ids=lambda examples: examples.name)
def test_example(examples, setup_doctest, monkeypatch):
    # Get the first example from the list of examples (there should be at least one)
    # raise UserWarning(PATH_SPIN_TEXTURE)
    for example in examples.examples:
        options = example.options
        if options:
            break
    # Get the options from the first example
    selected_path = None  # Initialize the selected path

    # Check if any path-related options are set
    if options:
        for key in options.keys():
            if key == PATH_SPIN_TEXTURE:
                # If the option is +PATH_SPIN_TEXTURE, we need to use the spin texture path
                selected_path = setup_doctest["path_spin_texture"]
                # raise UserWarning("using different path")
                break  # Exit loop after determining the path
            # You can add more checks here for other path flags if necessary

    # If no path was selected yet, fallback to the default path
    if selected_path is None:
        selected_path = setup_doctest["path"]

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
    monkeypatch.chdir(setup_doctest["path"])

    # Apply doctest options (ELLIPSIS, NORMALIZE_WHITESPACE)
    optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    runner = doctest.DocTestRunner(optionflags=optionflags)

    # Ensure calculation is properly set in the globals
    examples.globs["calculation"] = calculation

    # Run the doctest
    result = runner.run(examples)

    # Assert that there are no failed tests
    assert result.failed == 0
