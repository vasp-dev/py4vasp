# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import h5py
import numpy as np

from py4vasp._raw import data
from py4vasp._util import convert, parse


def structure(filename):
    contcar = CONTCAR(filename)
    return contcar.structure


def CONTCAR(filename):
    with open(filename, "r") as file:
        return parse.POSCAR(file.read())


def run_info_extension(filename):
    """Read every section of the /setup group VASP wrote into the h5 file.

    Subgroups of a section are flattened into keys like "functional_1/NAME". The data
    is read eagerly so that the file can be closed before returning."""
    with h5py.File(filename, "r") as h5f:
        setup = h5f.get("setup", {})
        sections = {
            name: _read_setup_section(group)
            for name, group in setup.items()
            if isinstance(group, h5py.Group)
        }
    return data.RunInfoExtension(sections=sections)


def _read_setup_section(group):
    section = data.SetupSection()
    group.visititems(lambda key, item: _add_to_section(section, key, item))
    return section


def _add_to_section(section, key, item):
    if not isinstance(item, h5py.Dataset):
        return
    section.values[key] = _read_setup_value(item)
    if "units" in item.attrs:
        section.units[key] = convert.text_to_string(item.attrs["units"])
    if "description" in item.attrs:
        section.descriptions[key] = convert.text_to_string(item.attrs["description"])


def _read_setup_value(dataset):
    value = dataset[()]
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "OS":
            strings = [convert.text_to_string(element) for element in value.flat]
            return np.array(strings).reshape(value.shape)
        return value
    if isinstance(value, bytes):
        return convert.text_to_string(value)
    return value.item() if isinstance(value, np.generic) else value
