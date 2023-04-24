# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Selector:
    maps: dict
    data: VaspData

    def __getitem__(self, key):
        index = self.maps[0][key]
        return self.data[index]
