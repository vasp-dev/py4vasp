# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import _demo, raw


def workfunction(direction):
    shape = (_demo.NUMBER_POINTS,)
    return raw.Workfunction(
        idipol=int(direction),
        distance=_demo.wrap_random_data(shape),
        average_potential=_demo.wrap_random_data(shape),
        vacuum_potential=_demo.wrap_random_data(shape=(2,)),
        reference_potential=_demo.bandgap.bandgap("nonpolarized"),
        fermi_energy=1.234,
    )
