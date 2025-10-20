# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def local_moment(selection):
    lmax = 3 if selection != "noncollinear" else 4
    number_components = _demo.number_components(selection)
    shape = (_demo.NUMBER_STEPS, number_components, _demo.NUMBER_ATOMS, lmax)
    moment = raw.LocalMoment(
        structure=_demo.structure.Fe3O4(),
        spin_moments=_demo.wrap_data(np.arange(np.prod(shape)).reshape(shape)),
    )
    if selection == "orbital_moments":
        remove_charge_and_s_component = moment.spin_moments[:, 1:, :, 1:]
        moment.orbital_moments = _demo.wrap_data(np.sqrt(remove_charge_and_s_component))
    return moment
