# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp.data import Potential, Structure


@pytest.fixture
def reference_potential(raw_data):
    def _reference_potential(
        selection, hartree_potential, ionic_potential, xc_potential
    ):
        raw_potential = raw_data.potential(
            selection=selection,
            hartree_potential=hartree_potential,
            ionic_potential=ionic_potential,
            xc_potential=xc_potential,
        )
        potential = Potential.from_data(raw_potential)
        potential.ref = types.SimpleNamespace()
        potential.ref.structure = Structure.from_data(raw_potential.structure).read()
        potential.ref.total_potential = raw_potential.total_potential
        if hartree_potential:
            potential.ref.hartree_potential = raw_potential.hartree_potential
        if ionic_potential:
            potential.ref.ionic_potential = raw_potential.ionic_potential
        if xc_potential:
            potential.ref.xc_potential = raw_potential.xc_potential
        return potential

    return _reference_potential
