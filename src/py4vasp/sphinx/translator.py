# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from docutils.nodes import NodeVisitor


class HugoTranslator(NodeVisitor):
    def __init__(self, document):
        self.document = document
        self.has_title = False
        self.section_level = 0
        self.content = []

    def unknown_departure(self, node):
        pass  # many markdown nodes do not require special handling for departure

    def visit_document(self, node):
        pass

    def visit_title(self, node):
        if not self.has_title:
            self.content.append(
                f"""\
+++
title = "{node.astext()}"
+++\n"""
            )
            self.has_title = True
        self.content.append(f"{self.section_level * '#'} ")

    def visit_section(self, node):
        self.section_level += 1

    def depart_section(self, node):
        self.section_level -= 1

    def visit_paragraph(self, node):
        self.content.append("\n")

    def visit_Text(self, node):
        self.content.append(f"{node.astext()}\n")
