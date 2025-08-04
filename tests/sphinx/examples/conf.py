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

napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = False
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True
napoleon_include_default_value = True
