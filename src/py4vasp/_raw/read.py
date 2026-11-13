# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import h5py
import numpy as np

from py4vasp import exception
from py4vasp._raw import data
from py4vasp._util import convert, parse


def structure(filename):
    contcar = CONTCAR(filename)
    return contcar.structure


def CONTCAR(filename):
    with open(filename, "r") as file:
        return parse.POSCAR(file.read())


def run_info_extension(filename):
    """Read every section of the /setup and /metadata groups VASP wrote into the file.

    The sections of /setup keep their name, the ones of /metadata get the prefix
    "metadata/", e.g. "metadata/build". Subgroups of a section are flattened into
    keys like "functional_1/NAME". The data is read eagerly so that the file can be
    closed before returning. Because access does not check the version of
    data_factory sources, this function raises an OutdatedVaspVersion error itself
    for files older than the version the schema requires for run_info_extension."""
    from py4vasp._raw.definition import DEFAULT_SOURCE, schema  # imports this module

    required = schema.sources["run_info_extension"][DEFAULT_SOURCE].required
    with h5py.File(filename, "r") as h5f:
        _check_version(h5f, required, filename)
        sections = {
            **_read_sections(h5f, "setup", prefix=""),
            **_read_sections(h5f, "metadata", prefix="metadata/"),
        }
    return data.RunInfoExtension(sections=sections)


def _read_sections(h5f, group_name, prefix):
    group = h5f.get(group_name, {})
    return {
        prefix + name: _read_setup_section(subgroup)
        for name, subgroup in group.items()
        if isinstance(subgroup, h5py.Group)
    }


def _check_version(h5f, required, filename):
    from py4vasp._raw.definition import schema  # definition imports this module

    try:
        version = data.Version(
            major=h5f[schema.version.major][()],
            minor=h5f[schema.version.minor][()],
            patch=h5f[schema.version.patch][()],
        )
    except KeyError:
        message = f"The settings in /setup are not available because {filename} does not specify a VASP version. They require at least {required}."
        raise exception.OutdatedVaspVersion(message) from None
    if version < required:
        message = f"The settings in /setup are not available in VASP {version}. They require at least {required}."
        raise exception.OutdatedVaspVersion(message)


def _read_setup_section(group):
    section = data.SetupSection()
    group.visititems(lambda key, item: _add_to_section(section, key, item))
    return section


def _add_to_section(section, key, item):
    if not isinstance(item, h5py.Dataset):
        return
    section.values[key] = _read_setup_value(item)
    if "units" in item.attrs:
        section.units[key] = convert.fortran_text_to_string(item.attrs["units"])
    if "description" in item.attrs:
        section.descriptions[key] = convert.fortran_text_to_string(
            item.attrs["description"]
        )


def _read_setup_value(dataset):
    value = dataset[()]
    if _is_logical(dataset):
        value = np.asarray(value).astype(np.bool_)
        return value if value.ndim > 0 else value.item()
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "OS":
            strings = [
                convert.fortran_text_to_string(element) for element in value.flat
            ]
            return np.array(strings).reshape(value.shape)
        return value
    if isinstance(value, bytes):
        return convert.fortran_text_to_string(value)
    return value.item() if isinstance(value, np.generic) else value


def _is_logical(dataset):
    "VASP stores logicals as integers 0/1 and marks them with the attribute dtype."
    dtype = dataset.attrs.get("dtype")
    return dtype is not None and convert.fortran_text_to_string(dtype) == "bool"
