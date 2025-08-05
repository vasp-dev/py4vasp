# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from docutils import nodes

from py4vasp._sphinx.builder import HugoBuilder


# Register the custom role for INCAR tags
def tag_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    url = f"https://vasp.at/wiki/index.php/{text}"
    node = nodes.reference(rawtext, text, refuri=url, **options)
    return [node], []


def setup(app):
    app.add_role("tag", tag_role)
    app.add_builder(HugoBuilder)
