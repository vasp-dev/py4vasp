# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import os
import sys

extensions = [
    "py4vasp._sphinx",
    "sphinx.ext.autodoc",
]

py4vasp_testing = True

# Configure autodoc to find the example module
sys.path.insert(0, os.path.abspath("."))
