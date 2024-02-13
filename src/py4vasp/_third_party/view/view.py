# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class View:
    number_ion_types: array
    ion_types: array
    lattice_vectors: array
    positions: array

    def to_ngl(self):
        pass
