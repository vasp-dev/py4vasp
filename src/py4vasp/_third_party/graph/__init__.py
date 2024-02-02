# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._config import VASP_COLORS
from py4vasp._util import import_

from .graph import Graph
from .mixin import Mixin
from .plot import plot
from .series import Series

go = import_.optional("plotly.graph_objects")
pio = import_.optional("plotly.io")

if import_.is_imported(go) and import_.is_imported(pio):
    axis_format = {"showexponent": "all", "exponentformat": "power"}
    layout = {"colorway": VASP_COLORS, "xaxis": axis_format, "yaxis": axis_format}
    pio.templates["vasp"] = go.layout.Template(layout=layout)
    pio.templates.default = "ggplot2+vasp"
