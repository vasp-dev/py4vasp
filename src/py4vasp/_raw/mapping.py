# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import annotations

import dataclasses
from collections import abc


@dataclasses.dataclass
class Mapping(abc.Mapping):
    valid_indices: Sequence

    def __len__(self):
        return len(self.valid_indices)

    def __iter__(self):
        return iter(self.valid_indices)

    def __getitem__(self, key):
        index = self.valid_indices.index(key)
        elements = {
            key: value[index] if isinstance(value, list) else value
            for key, value in self._as_dict().items()
            if key != "valid_indices"
        }
        return dataclasses.replace(self, valid_indices=[key], **elements)

    def _as_dict(self):
        # shallow copy of dataclass to dictionary
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if getattr(self, field.name) is not None
        }
