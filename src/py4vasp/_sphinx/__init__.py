# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import os
from enum import member
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList
from jinja2 import Template

from py4vasp import _calculation
from py4vasp._sphinx.builder import HugoBuilder


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


# Register the custom role for INCAR tags
def tag_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    url = f"https://vasp.at/wiki/index.php/{text}"
    node = nodes.reference(rawtext, text, refuri=url, **options)
    return [node], []


def generate_quantity_docs(app):
    if app.config.py4vasp_testing:
        return
    included_folder = app.srcdir / "calculation"
    included_folder.mkdir(exist_ok=True, parents=True)
    hidden_folder = app.srcdir / "hidden" / "calculation"
    hidden_folder.mkdir(exist_ok=True, parents=True)
    for quantity in _calculation.QUANTITIES:
        if not should_write(app, included_folder, quantity):
            continue
        write_docstring(included_folder, quantity)
        write_hidden_docstring(hidden_folder, quantity)
    for group, members in _calculation.GROUPS.items():
        for member in members:
            quantity = f"{group}_{member}"
            if not should_write(app, included_folder, quantity):
                continue
            write_docstring(included_folder, quantity, title=f"{group}.{member}")
            # write_hidden_docstring(hidden_folder, quantity)


def should_write(app, folder, quantity):
    if quantity.startswith("_"):
        return False
    outfile = folder / f"{quantity}.rst"
    if outfile.exists():
        infile = app.srcdir / f"../src/py4vasp/_calculation/{quantity}.py"
        infile_mtime = os.path.getmtime(infile)
        outfile_mtime = os.path.getmtime(outfile)
        if outfile_mtime >= infile_mtime:
            return False
    return True


def write_docstring(folder, quantity, title=None):
    if title is None:
        title = quantity
    outfile = folder / f"{quantity}.rst"
    with open(outfile, "w", encoding="utf-8") as file:
        file.write(f"{title}\n")
        file.write("=" * len(title) + "\n\n")
        file.write(f".. automodule:: py4vasp._calculation.{quantity}\n")
        file.write("   :members:\n")
        file.write("   :inherited-members:\n")


def write_hidden_docstring(folder, quantity):
    outfile = folder / f"{quantity}.rst"
    with open(outfile, "w", encoding="utf-8") as file:
        file.write(f"{quantity}\n")
        file.write("=" * len(quantity) + "\n\n")
        file.write(f".. autoproperty:: py4vasp.Calculation.{quantity}\n")


def on_build_finished(app, exception):
    """Hook called when Sphinx build is finished.

    If the build was successful and used the Hugo builder, assign weights
    to the generated markdown files.
    """
    # Only run if build was successful
    if exception is not None:
        return

    # Only run for Hugo builder
    if app.builder.name != "hugo":
        return

    # Import here to avoid circular dependency
    from py4vasp._sphinx.assign_hugo_weights import assign_weights

    # Get the Hugo output directory
    hugo_dir = Path(app.outdir) / "hugo"

    if hugo_dir.exists():
        print("\nAssigning Hugo weights to markdown files...")
        assign_weights(hugo_dir)


def setup(app):
    app.add_builder(HugoBuilder)
    app.add_config_value("py4vasp_testing", False, "env")
    app.add_directive("jinja", JinjaDirective)
    app.add_role("tag", tag_role)
    app.connect("builder-inited", generate_quantity_docs)
    app.connect("build-finished", on_build_finished)
    app.connect("build-finished", on_build_finished)
