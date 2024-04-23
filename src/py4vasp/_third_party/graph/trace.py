# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from abc import ABC, abstractmethod


class Trace(ABC):
    """Defines a base class with all methods that need to be implemented for Graph to
    work as intended"""

    @abstractmethod
    def to_plotly(self):
        """Use yield to generate one or more plotly traces. Each returned element should
        be a tuple (trace, dict) where the trace can be used as data for plotly and the
        options modify the generation of the figure."""
