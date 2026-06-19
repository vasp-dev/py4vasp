# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pathlib

import h5py
import numpy as np

from py4vasp import raw
from py4vasp._raw.definition import DEFAULT_FILE
from py4vasp._raw.write import write


def test_write(tmp_path, raw_data, Assert):
    filename = tmp_path / DEFAULT_FILE
    raw_structure = raw_data.structure("Sr2TiO4")
    with h5py.File(filename, "w") as h5f:
        write(h5f, raw_structure)
    with raw.access("structure", path=tmp_path) as structure:
        Assert.same_raw_structure(raw_structure, structure)


def test_write_selection(tmp_path, raw_data, Assert):
    filename = tmp_path / DEFAULT_FILE
    raw_structure = raw_data.structure("Sr2TiO4")
    with h5py.File(filename, "w") as h5f:
        write(h5f, raw.Version(99, 99, 99))
        write(h5f, raw_structure, selection="final")
    with raw.access("structure", path=tmp_path, selection="final") as structure:
        Assert.same_raw_structure(raw_structure, structure)


def test_write_encodes_unicode_strings(tmp_path):
    # h5py cannot serialize numpy unicode arrays (dtype kind "U"); the writer must encode
    # them as byte strings, matching how VASP stores strings (e.g. effective_coulomb's
    # str spin_labels). Numeric fields are unaffected.
    filename = tmp_path / DEFAULT_FILE
    stoichiometry = raw.Stoichiometry(
        number_ion_types=np.array([2, 1, 4]),
        ion_types=np.array(["Sr", "Ti", "O"]),  # unicode, not bytes
    )
    with h5py.File(filename, "w") as h5f:
        write(h5f, stoichiometry)
    with raw.access("stoichiometry", path=tmp_path) as actual:
        assert np.array_equal(actual.ion_types, np.array([b"Sr", b"Ti", b"O"]))
        assert np.array_equal(actual.number_ion_types, [2, 1, 4])
