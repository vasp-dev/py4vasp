# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

# dummy sphinx conf.py file
import os
import sys

# Configure autodoc to find the example module
sys.path.insert(0, os.path.abspath("."))

extensions = [
    "py4vasp._sphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autosummary_generate_overwrite = True
templates_path = [os.path.abspath("_templates")]
autosummary_ignore_module_all = False

# Debug: print to verify template path
print(f"DEBUG: templates_path = {templates_path}")

# Keep type aliases unexpanded in documentation
autodoc_type_aliases = {
    'ArrayLike': 'ArrayLike',
    'float | _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes]': 'ArrayLike',
}

napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    'ArrayLike': 'ArrayLike',
    'float | _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes]': 'ArrayLike',
}
napoleon_attr_annotations = True
napoleon_include_default_value = True
