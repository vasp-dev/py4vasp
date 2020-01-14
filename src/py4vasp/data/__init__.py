from .band import Band
from .dos import Dos

import plotly.io as pio
import cufflinks as cf

pio.templates.default = "ggplot2"
cf.go_offline()
cf.set_config_file(theme="ggplot")
