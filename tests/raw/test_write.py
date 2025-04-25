# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import h5py

from py4vasp import raw
from py4vasp._raw.definition import DEFAULT_FILE
from py4vasp._raw.write import write


def test_write(tmp_path, raw_data, Assert):
    filename = tmp_path / DEFAULT_FILE
    raw_structure = raw_data.structure("Sr2TiO4")
    with h5py.File(filename, "w") as h5f:
        h5f = write(h5f, raw_structure)
    with raw.access("structure", path=tmp_path) as structure:
        Assert.same_raw_structure(raw_structure, structure)
