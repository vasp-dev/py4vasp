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
    pio.templates["vasp"] = go.layout.Template(layout={"colorway": VASP_COLORS})
    pio.templates.default = "ggplot2+vasp"
