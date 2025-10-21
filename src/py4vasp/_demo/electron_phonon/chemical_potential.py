# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def chemical_potential(selection="carrier_den"):
    seed = 26826821
    temperature_mesh = np.linspace(0, 500, _demo.NUMBER_TEMPERATURES)
    return raw.ElectronPhononChemicalPotential(
        fermi_energy=np.random.randn(),
        carrier_density=_demo.wrap_random_data(
            [_demo.NUMBER_CHEMICAL_POTENTIALS, _demo.NUMBER_TEMPERATURES]
        ),
        temperatures=_demo.wrap_data(temperature_mesh),
        chemical_potential=_demo.wrap_random_data(
            [_demo.NUMBER_CHEMICAL_POTENTIALS, _demo.NUMBER_TEMPERATURES]
        ),
        carrier_per_cell=_demo.wrap_random_data(
            [_demo.NUMBER_CHEMICAL_POTENTIALS],
            selection == "carrier_per_cell",
            seed=seed,
        ),
        mu=_demo.wrap_random_data(
            [_demo.NUMBER_CHEMICAL_POTENTIALS], selection == "mu", seed=seed
        ),
        carrier_den=_demo.wrap_random_data(
            [_demo.NUMBER_CHEMICAL_POTENTIALS], selection == "carrier_den", seed=seed
        ),
    )
