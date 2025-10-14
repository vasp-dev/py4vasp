# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def Sr2TiO4():
    labels = ("total", "Sr~Sr", "Sr~Ti", "Sr~O", "Ti~Ti", "Ti~O", "O~O")
    shape = (_demo.NUMBER_STEPS, len(labels), _demo.NUMBER_POINTS)
    data = np.arange(np.prod(shape)).reshape(shape)
    return raw.PairCorrelation(
        distances=np.arange(_demo.NUMBER_POINTS), function=data, labels=labels
    )
