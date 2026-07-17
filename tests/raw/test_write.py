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


def test_write_skips_field_absent_from_source_schema(tmp_path, Assert):
    # The ionic() demo sets a q_point, but the "ion" source schema defines no q_point
    # path. The writer must skip that field (this source simply does not store it)
    # instead of crashing when it tries to use the unset schema target as an HDF5 key.
    from py4vasp._demo.dielectric_function import ionic

    filename = tmp_path / DEFAULT_FILE
    raw_dielectric = ionic()
    assert not raw_dielectric.q_point.is_none()  # the demo does set q_point
    with h5py.File(filename, "w") as h5f:
        write(h5f, raw.Version(99, 99, 99))
        write(h5f, raw_dielectric, selection="ion")  # must not raise
    with raw.access("dielectric_function", path=tmp_path, selection="ion") as actual:
        Assert.allclose(actual.energies, raw_dielectric.energies)
        assert actual.q_point.is_none()  # not stored by the "ion" source


def test_write_skips_absent_mapping_entries(tmp_path):
    # A Mapping field is a per-index list whose entries may be absent (VaspData(None))
    # for the chosen selection. electron_phonon self_energy with "CRTA" has nbands_sum
    # and delta as VaspData(None) for every sample; those must be skipped, not written
    # (serializing VaspData(None) raises NoData), while present fields are written.
    from py4vasp._demo.electron_phonon import self_energy

    filename = tmp_path / DEFAULT_FILE
    raw_data = self_energy.self_energy("CRTA")
    with h5py.File(filename, "w") as h5f:
        write(h5f, raw_data)
    with h5py.File(filename, "r") as h5f:
        group = "results/electron_phonon/electrons/self_energy_0"
        assert f"{group}/selfen_fan" in h5f
        assert f"{group}/nbands_sum" not in h5f
        assert f"{group}/delta" not in h5f
