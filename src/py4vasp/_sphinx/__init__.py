# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._sphinx.builder import HugoBuilder


def setup(app):
    app.add_builder(HugoBuilder)
