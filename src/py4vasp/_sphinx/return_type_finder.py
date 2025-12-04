# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from docutils.nodes import NodeVisitor, SkipNode


def _extract_type_text(node) -> str:
    """Extract type text from a node, handling TypeAliasForwardRef properly.
    
    The issue is that node.astext() returns the string representation of 
    TypeAliasForwardRef objects like "TypeAliasForwardRef('ArrayLike')" 
    instead of just "ArrayLike". We need to extract text from child Text nodes
    and then parse out the actual type name from the TypeAliasForwardRef wrapper.
    """
    import re
    from docutils.nodes import Text
    text_parts = []
    for child in node.findall(Text):
        text_parts.append(str(child))
    text = ''.join(text_parts)
    # Replace TypeAliasForwardRef('TypeName') with just TypeName
    text = re.sub(r"TypeAliasForwardRef\('([^']+)'\)", r'\1', text)
    return text


class ReturnTypeFinder(NodeVisitor):
    """A docutils NodeVisitor that finds return types in Sphinx document trees."""

    def __init__(self, document):
        super().__init__(document)
        self._return_type = None
        self._in_returns_type_field = False

    def __str__(self):
        return "\n".join(self.lines) + "\n"

    def find_return_type(self, node) -> str:
        """Retrieves the return type from a desc_signature node.

        Parameters
        ----------
        node : docutils.nodes.Node
            The desc_signature node to retrieve the return type from.

        Returns
        -------
        str
            The return type of the function or method.
        """
        if node.__class__.__name__ != "desc_signature":
            raise UserWarning(
                "Node passed to get_return_type is not a desc_signature node. Return type not retrieved."
            )
        else:
            self._return_type = ""
            node.parent.walkabout(self)
        return self._return_type

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
        pass

    def visit_desc_signature(self, node):
        pass

    def visit_desc_content(self, node):
        pass

    def visit_field_list(self, node):
        pass

    def visit_field(self, node):
        pass

    def visit_desc_returns(self, node):
        if not (self._return_type):
            self._return_type = (
                _extract_type_text(node)
                .lstrip(" -> ")
                .strip()
                .replace("` or `", " or ")
                .replace(" or ", " | ")
            )
        raise SkipNode

    def visit_field_name(self, node):
        if node.astext().lower() == "return type":
            self._in_returns_type_field = True
        raise SkipNode

    def visit_field_body(self, node):
        if self._in_returns_type_field:
            if not (self._return_type):
                self._return_type = (
                    _extract_type_text(node)
                    .strip()
                    .replace("` or `", " or ")
                    .replace(" or ", " | ")
                )
        raise SkipNode

    def depart_field(self, node):
        self._in_returns_type_field = False
        pass
