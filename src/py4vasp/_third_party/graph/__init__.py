# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import plotly.graph_objects as go
import plotly.io as pio

from py4vasp._config import VASP_COLORS

from .graph import Graph
from .mixin import Mixin
from .plot import plot
from .series import Series

pio.templates["vasp"] = go.layout.Template(layout={"colorway": VASP_COLORS})
pio.templates.default = "ggplot2+vasp"
