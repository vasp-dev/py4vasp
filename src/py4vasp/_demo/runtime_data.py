# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import _demo, raw


def runtime_data(system_name: str = None):
    return raw.RuntimeData(
        vasp_version=raw.Version(99, 99, 99),
    )
