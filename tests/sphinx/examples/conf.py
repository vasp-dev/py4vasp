# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

# dummy sphinx conf.py file
extensions = [
    "py4vasp._sphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon", 
]

# Configure autodoc to find the example module
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
