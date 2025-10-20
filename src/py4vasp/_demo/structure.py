# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def BN():
    return raw.Structure(
        raw.Stoichiometry(number_ion_types=[1, 1], ion_types=["B", "N"]),
        raw.Cell(
            lattice_vectors=np.array(
                [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
            ),
            scale=raw.VaspData(3.63),
        ),
        positions=np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
    )


def Ca3AsBr3():
    positions = [
        [0.5, 0.0, 0.0],  # Ca_1
        [0.0, 0.5, 0.0],  # Ca_2
        [0.0, 0.0, 0.0],  # As
        [0.0, 0.5, 0.5],  # Br_1
        [0.0, 0.0, 0.5],  # Ca_3
        [0.5, 0.0, 0.5],  # Br_2
        [0.5, 0.5, 0.0],  # Br_3
    ]
    return raw.Structure(
        stoichiometry=_demo.stoichiometry.Ca3AsBr3(),
        cell=_demo.cell.Ca3AsBr3(),
        positions=_demo.wrap_data(positions),
    )


def CaAs3_110():
    positions = [
        [0.20000458, 0.51381288, 0.73110298],
        [0.79999542, 0.48618711, 0.66008269],
        [0.20000458, 0.51381288, 0.93991731],
        [0.70000458, 0.01381289, 0.83551014],
        [0.79999542, 0.48618711, 0.86889702],
        [0.29999541, 0.98618712, 0.76448986],
        [0.08920607, 0.11201309, 0.67393241],
        [0.91079393, 0.88798690, 0.71725325],
        [0.57346071, 0.83596581, 0.70010722],
        [0.42653929, 0.16403419, 0.69107845],
        [0.72035614, 0.40406032, 0.73436505],
        [0.27964386, 0.59593968, 0.65682062],
        [0.08920607, 0.11201309, 0.88274675],
        [0.58920607, 0.61201310, 0.77833958],
        [0.91079393, 0.88798690, 0.92606759],
        [0.41079393, 0.38798690, 0.82166042],
        [0.57346071, 0.83596581, 0.90892155],
        [0.07346071, 0.33596581, 0.80451438],
        [0.42653929, 0.16403419, 0.89989278],
        [0.92653929, 0.66403419, 0.79548562],
        [0.72035614, 0.40406032, 0.94317938],
        [0.22035614, 0.90406032, 0.83877221],
        [0.27964386, 0.59593968, 0.86563495],
        [0.77964386, 0.09593968, 0.76122779],
    ]
    return raw.Structure(
        stoichiometry=_demo.stoichiometry.CaAs3_110(),
        cell=_demo.cell.CaAs3_110(),
        positions=raw.VaspData(positions),
    )


def SrTiO3():
    return raw.Structure(
        raw.Stoichiometry(number_ion_types=[1, 1, 3], ion_types=["Sr", "Ti", "O"]),
        raw.Cell(lattice_vectors=np.eye(3), scale=raw.VaspData(4.0)),
        positions=np.array(
            [
                [0, 0, 0],
                [0.5, 0.5, 0.5],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ]
        ),
    )


def Sr2TiO4(has_ion_types=True):
    repetitions = (_demo.NUMBER_STEPS, 1, 1)
    positions = [
        [0.64529, 0.64529, 0.0],
        [0.35471, 0.35471, 0.0],
        [0.00000, 0.00000, 0.0],
        [0.84178, 0.84178, 0.0],
        [0.15823, 0.15823, 0.0],
        [0.50000, 0.00000, 0.5],
        [0.00000, 0.50000, 0.5],
    ]
    return raw.Structure(
        stoichiometry=_demo.stoichiometry.Sr2TiO4(has_ion_types),
        cell=_demo.cell.Sr2TiO4(),
        positions=np.tile(positions, repetitions),
    )


def Fe3O4():
    positions = [
        [0.00000, 0.0, 0.00000],
        [0.50000, 0.0, 0.50000],
        [0.00000, 0.5, 0.50000],
        [0.78745, 0.0, 0.28152],
        [0.26310, 0.5, 0.27611],
        [0.21255, 0.0, 0.71848],
        [0.73690, 0.5, 0.72389],
    ]
    shift = np.linspace(-0.02, 0.01, _demo.NUMBER_STEPS)
    return raw.Structure(
        stoichiometry=_demo.stoichiometry.Fe3O4(),
        cell=_demo.cell.Fe3O4(),
        positions=np.add.outer(shift, positions),
    )


def Graphite():
    positions = [
        [0.00000000, 0.00000000, 0.00000000],
        [0.33333333, 0.66666667, 0.00000000],
        [0.33333333, 0.66666667, 0.15031929],
        [0.66666667, 0.33333333, 0.15031929],
        [0.00000000, 0.00000000, 0.30063858],
        [0.33333333, 0.66666667, 0.30063858],
        [0.33333333, 0.66666667, 0.45095787],
        [0.66666667, 0.33333333, 0.45095787],
        [0.00000000, 0.00000000, 0.60127716],
        [0.33333333, 0.66666667, 0.60127716],
    ]
    return raw.Structure(
        stoichiometry=_demo.stoichiometry.Graphite(),
        cell=_demo.cell.Graphite(),
        positions=raw.VaspData(positions),
    )


def Ni100():
    positions = [
        [0.00000000, 0.00000000, 0.00000000],
        [0.50000000, 0.10000000, 0.50000000],
        [0.00000000, 0.20000000, 0.00000000],
        [0.50000000, 0.30000000, 0.50000000],
        [0.00000000, 0.40000000, 0.00000000],
    ]
    return raw.Structure(
        stoichiometry=_demo.stoichiometry.Ni100(),
        cell=_demo.cell.Ni100(),
        positions=raw.VaspData(positions),
    )


def ZnS():
    return raw.Structure(
        raw.Stoichiometry(number_ion_types=[2, 2], ion_types=["Zn", "S"]),
        raw.Cell(
            lattice_vectors=np.array([[1.9, -3.3, 0.0], [1.9, 3.3, 0.0], [0, 0, 6.2]]),
            scale=raw.VaspData(1.0),
        ),
        positions=np.array(
            [
                [1 / 3, 2 / 3, 0.0],
                [2 / 3, 1 / 3, 0.5],
                [1 / 3, 2 / 3, 0.375],
                [2 / 3, 1 / 3, 0.875],
            ]
        ),
    )
