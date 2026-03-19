# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses

from py4vasp._raw.definition import DEFAULT_SOURCE, schema
from py4vasp._raw.mapping import Mapping
from py4vasp._raw.schema import Length, Link
from py4vasp._util import check, convert


def write(h5f, raw_data, *, selection=None):
    quantity = convert.quantity_name(raw_data.__class__.__name__)
    source = schema.sources[quantity][selection or DEFAULT_SOURCE]
    valid_indices = None
    if isinstance(raw_data, Mapping):
        valid_indices = raw_data.valid_indices
    for field in dataclasses.fields(source.data):
        target = getattr(source.data, field.name)
        data = getattr(raw_data, field.name)
        _write_dataset(h5f, target, data, valid_indices)


def _write_dataset(h5f, target, data, valid_indices=None):
    if isinstance(target, Link):
        write(h5f, data, selection=target.source)
    elif check.is_none(data) or isinstance(target, Length) or target in h5f:
        return
    elif (
        valid_indices is not None and "{" in target and isinstance(data, (list, tuple))
    ):
        # Handle template paths for Mapping types with multiple indices
        for index, item in zip(valid_indices, data):
            expanded_target = target.format(index)
            h5f[expanded_target] = item
    else:
        # TODO: deal with type conversion for strings
        h5f[target] = data
