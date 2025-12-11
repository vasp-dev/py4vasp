# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from pathlib import Path

from docutils import nodes

from py4vasp._sphinx.builder import HugoBuilder


# Register the custom role for INCAR tags
def tag_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    url = f"https://vasp.at/wiki/index.php/{text}"
    node = nodes.reference(rawtext, text, refuri=url, **options)
    return [node], []


def on_build_finished(app, exception):
    """Hook called when Sphinx build is finished.
    
    If the build was successful and used the Hugo builder, assign weights
    to the generated markdown files.
    """
    # Only run if build was successful
    if exception is not None:
        return
    
    # Only run for Hugo builder
    if app.builder.name != 'hugo':
        return
    
    # Import here to avoid circular dependency
    from py4vasp._sphinx.assign_hugo_weights import assign_weights
    
    # Get the Hugo output directory
    hugo_dir = Path(app.outdir) / "hugo"
    
    if hugo_dir.exists():
        print("\nAssigning Hugo weights to markdown files...")
        assign_weights(hugo_dir)


def setup(app):
    app.add_role("tag", tag_role)
    app.add_builder(HugoBuilder)
    app.connect('build-finished', on_build_finished)
