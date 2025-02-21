# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import annotations

import dataclasses
from collections import abc

from py4vasp import exception
from py4vasp._util import suggest


@dataclasses.dataclass
class Mapping(abc.Mapping):
    valid_indices: Sequence

    def __len__(self):
        return len(self.valid_indices)

    def __iter__(self):
        return iter(self.valid_indices)

    def __getitem__(self, key):
        index = self.try_to_find_key_in_valid_indices(key)
        elements = {
            key: value[index] if isinstance(value, list) else value
            for key, value in self._as_dict().items()
            if key != "valid_indices"
        }
        return dataclasses.replace(self, valid_indices=[key], **elements)

    def try_to_find_key_in_valid_indices(self, key):
        try:
            return self.valid_indices.index(key)
        except ValueError:
            did_you_mean = suggest.did_you_mean(key, self.valid_indices)
            message = f"""\
Could not find the selection "{key}" in the valid selections. {did_you_mean}\
Please check for possible spelling errors. The following selections are possible: \
{", ".join(f'"{index}"' for index in self.valid_indices)}."""
            raise exception.IncorrectUsage(message)

    def _as_dict(self):
        # shallow copy of dataclass to dictionary
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if getattr(self, field.name) is not None
        }
