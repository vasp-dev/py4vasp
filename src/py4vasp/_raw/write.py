# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses

from py4vasp._raw.definition import DEFAULT_SOURCE, schema
from py4vasp._raw.schema import Length, Link
from py4vasp._util import check, convert


def write(h5f, raw_data, *, selection=None):
    quantity = convert.quantity_name(raw_data.__class__.__name__)
    source = schema.sources[quantity][selection or DEFAULT_SOURCE]
    for field in dataclasses.fields(source.data):
        target = getattr(source.data, field.name)
        data = getattr(raw_data, field.name)
        _write_dataset(h5f, target, data)


def _write_dataset(h5f, target, data):
    if isinstance(target, Link):
        write(h5f, data, selection=target.source)
    elif check.is_none(data) or isinstance(target, Length) or target in h5f:
        return
    else:
        h5f[target] = data
