# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def MD(randomize: bool = False):
    labels = (
        "ion-electron   TOTEN",
        "kinetic energy EKIN",
        "kin. lattice   EKIN_LAT",
        "temperature    TEIN",
        "nose potential ES",
        "nose kinetic   EPS",
        "total energy   ETOTAL",
    )
    return _create_energy(labels, randomize=randomize)


def relax(randomize: bool = False):
    labels = (
        "free energy    TOTEN   ",
        "energy without entropy ",
        "energy(sigma->0)       ",
    )
    return _create_energy(labels, randomize=randomize)


def afqmc():
    labels = (
        "step            STEP    ",
        "One el. energy  E1      ",
        "Hartree energy  -DENC   ",
        "exchange        EXHF    ",
        "free energy     TOTEN   ",
        "free energy cap TOTENCAP",
        "weight          WEIGHT  ",
    )
    return _create_energy(labels)


def _create_energy(labels, randomize: bool = False):
    labels = np.array(labels, dtype="S")
    shape = (_demo.NUMBER_STEPS, len(labels))
    if randomize:
        return raw.Energy(labels=labels, values=np.random.random(shape))
    else:
        return raw.Energy(
            labels=labels, values=np.arange(np.prod(shape)).reshape(shape)
        )
