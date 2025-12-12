# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------

project = "py4vasp"
copyright = "2025, VASP Software GmbH"
author = "VASP Software GmbH"

# The full version, including alpha/beta/rc tags
release = "0.11.0"
root_doc = "_index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]
# Settings for autosummary
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_ignore_module_all = False

# Keep type aliases unexpanded in documentation
autodoc_type_aliases = {
    "ArrayLike": "ArrayLike",
    "Buffer | _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes]": "ArrayLike",
    "float | _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes]": "ArrayLike",
}
# Autodoc configuration for inherited members
autodoc_inherit_docstrings = True
autodoc_default_options = {
    "inherited-members": True,
    "undoc-members": True,
}
# Preserve default argument values as written in source code
autodoc_preserve_defaults = True

# Configuration of Napoleon extension
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "ArrayLike": "ArrayLike",
    "Buffer | _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes]": "ArrayLike",
    "float | _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes]": "ArrayLike",
}
napoleon_attr_annotations = True
napoleon_include_default_value = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# # default theme
# html_theme = "nature"

# a minimal theme for the website
html_theme = "basic"
html_show_sphinx = False
html_show_copyright = False
html_domain_indices = False
html_use_index = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# remove common py4vasp prefix from index
modindex_common_prefix = ["py4vasp."]
