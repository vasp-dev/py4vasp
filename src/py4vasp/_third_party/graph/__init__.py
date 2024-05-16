# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy

from py4vasp._config import VASP_BLUE, VASP_COLORS, VASP_GRAY, VASP_RED
from py4vasp._util import import_

from .contour import Contour
from .graph import Graph
from .mixin import Mixin
from .plot import plot
from .series import Series

go = import_.optional("plotly.graph_objects")
pio = import_.optional("plotly.io")

if import_.is_imported(go) and import_.is_imported(pio):
    axis_format = {"showexponent": "all", "exponentformat": "power"}
    contour = copy.copy(pio.templates["ggplot2"].data.contour[0])
    contour.colorscale = [[0, VASP_BLUE], [0.5, VASP_GRAY], [1, VASP_RED]]
    data = {"contour": (contour,)}
    layout = {"colorway": VASP_COLORS, "xaxis": axis_format, "yaxis": axis_format}
    pio.templates["vasp"] = go.layout.Template(data=data, layout=layout)
    pio.templates["ggplot2"].layout.shapedefaults = {}
    pio.templates.default = "ggplot2+vasp"
