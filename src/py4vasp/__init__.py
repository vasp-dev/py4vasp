# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from .calculation import Calculation
from py4vasp._third_party.interactive import set_error_handling
from py4vasp._third_party.graph import plot

__version__ = "0.5.1"
set_error_handling("Minimal")
