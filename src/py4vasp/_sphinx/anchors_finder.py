# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from docutils.nodes import NodeVisitor, SkipNode


class AnchorsFinder(NodeVisitor):
    """A docutils NodeVisitor that finds signature anchor IDs (like script_name.Class_name.method_name) in Sphinx document trees."""

    def __init__(self, document):
        super().__init__(document)
        self._name = ""
        self._addname = ""
        self._objtype = ""
        self._domain = ""

    def __str__(self):
        return "\n".join(self.lines) + "\n"

    def find_anchors(self, node) -> list:
        if node.__class__.__name__ != "desc":
            raise UserWarning(
                "Node passed to find_anchors is not a desc node. Anchors not retrieved."
            )
        else:
            self._name = ""
            self._addname = ""
            self._objtype = ""
            self._domain = ""
            node.walkabout(self)
        return (self._name, self._addname, self._objtype, self._domain)

    def unknown_visit(self, node):
        raise SkipNode

    def unknown_departure(self, node):
        """Handle departure from nodes that don't have specific depart methods.

        Most Markdown constructs don't require closing syntax, so we provide this
        no-op handler to avoid errors when the visitor tries to call depart methods
        that don't exist.
        """
        pass

    def visit_desc(self, node):
        self._domain = getattr(node, "attributes", {}).get("domain", "")
        self._objtype = getattr(node, "attributes", {}).get("objtype", "")

    def visit_desc_signature(self, node): 
        pass

    def visit_desc_name(self, node): self._name = node.astext().strip(". ").strip(" .")

    def visit_desc_addname(self, node): self._addname = node.astext().strip(". ").strip(" .")