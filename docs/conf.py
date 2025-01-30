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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "py4vasp"
copyright = "2024, VASP Software GmbH"
author = "VASP Software GmbH"

# The full version, including alpha/beta/rc tags
release = "0.10.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.napoleon", "sphinx_automodapi.automodapi"]
automodapi_inheritance_diagram = False
autosummary_ignore_module_all = False

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


# -- Custom extension of Sphinx ----------------------------------------------
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList
from jinja2 import Template
from py4vasp import _calculation, _util


# defines an INCAR tag
def tag_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    url = f"https://www.vasp.at/wiki/index.php/{text}"
    node = nodes.reference(rawtext, text, refuri=url, **options)
    return [node], []


class JinjaDirective(Directive):
    has_content = True

    def run(self):
        template_string = "\n".join(self.content)
        template = Template(template_string)
        rendered_content = template.render(calculation=_calculation)
        filename = self.state.document.current_source
        view_list = ViewList(rendered_content.split("\n"), filename)
        node = nodes.Element()
        self.state.nested_parse(view_list, 0, node)
        return node.children


def setup(app):
    app.add_directive("jinja", JinjaDirective)
    app.add_role("tag", tag_role)
    return
