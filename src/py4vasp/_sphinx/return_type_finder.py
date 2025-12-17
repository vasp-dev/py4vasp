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
    text = "".join(text_parts)
    # Replace TypeAliasForwardRef('TypeName') with just TypeName
    text = re.sub(r"TypeAliasForwardRef\('([^']+)'\)", r"\1", text)
    return text


class ReturnTypeFinder(NodeVisitor):
    """A docutils NodeVisitor that finds return types in Sphinx document trees.

    Extracts return type from both the function signature annotation and the
    Returns field in the docstring, with signature taking priority.
    """

    def __init__(self, document):
        super().__init__(document)
        self._signature_return_type = None
        self._docstring_return_type = None
        self._in_returns_field = False

    def __str__(self):
        return "\n".join(self.lines) + "\n"

    def find_return_type(self, node) -> tuple[str, str]:
        """Retrieves the return type from a desc_signature node.

        Extracts return type from both signature annotation and docstring Returns field.

        Parameters
        ----------
        node : docutils.nodes.Node
            The desc_signature node to retrieve the return type from.

        Returns
        -------
        tuple[str, str]
            A tuple of (signature_return_type, docstring_return_type).
            Either can be None or empty string if not present.
        """
        if node.__class__.__name__ != "desc_signature":
            raise UserWarning(
                "Node passed to find_return_type is not a desc_signature node. Return type not retrieved."
            )
        else:
            self._signature_return_type = ""
            self._docstring_return_type = ""
            node.parent.walkabout(self)
        return self._signature_return_type, self._docstring_return_type

    def unknown_visit(self, node):
        # Don't skip unknown nodes, let traversal continue
        pass

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
        # Check if this is a Returns or Return type field
        for child in node.children:
            if child.__class__.__name__ == "field_name":
                field_name = child.astext().strip().lower()
                if field_name == "return type":
                    # Extract the return type from this field's body
                    for field_child in node.children:
                        if field_child.__class__.__name__ == "field_body":
                            # The type is in a paragraph, possibly in a literal node
                            type_text = field_child.astext().strip()
                            if type_text and not self._docstring_return_type:
                                self._set_docstring_return_type_if_applicable(type_text)
                elif "return" in field_name and field_name != "return type":
                    self._in_returns_field = True
                break
        pass

    def visit_desc_returns(self, node):
        if not (self._signature_return_type):
            self._signature_return_type = (
                _extract_type_text(node)
                .lstrip(" -> ")
                .strip()
                .replace("` or `", " or ")
                .replace(" or ", " | ")
            )
        raise SkipNode

    def visit_section(self, node):
        # Check if this is a Returns section
        for child in node.children:
            if child.__class__.__name__ == "title":
                title_text = child.astext().lower()
                if "return" in title_text:
                    self._in_returns_field = True
                break

    def depart_section(self, node):
        # Reset the flag when leaving any section
        self._in_returns_field = False

    def visit_definition_list(self, node):
        # NumPy-style Returns sections are converted to definition lists
        pass

    def visit_definition_list_item(self, node):
        if self._in_returns_field and not self._docstring_return_type:
            # The term contains the type, definition contains the description
            for child in node.children:
                if child.__class__.__name__ == "term":
                    term_text = child.astext().strip().strip("`")
                    self._set_docstring_return_type_if_applicable(term_text)
                    break

    def _set_docstring_return_type_if_applicable(self, term_text: str):
        """Set the docstring return type if the term text looks like a type."""
        if not self._docstring_return_type:
            # if it looks like a type, set it
            # heuristic: types are usually short, single words or simple unions
            # heuristic: types are given on a separate line from their description
            # heuristic: a description might not be available
            # heuristic: the entire line might be a description
            separated = term_text.split("\n")
            if (len(separated[0].strip().split(" "))) == 1:
                self._docstring_return_type = separated[0].strip()
            elif (" | " in separated[0] or " or " in separated[0]) and all(
                len(part.strip().split(" ")) == 1
                for part in separated[0].replace(" or ", " | ").split(" | ")
            ):
                self._docstring_return_type = (
                    separated[0].strip().replace(" or ", " | ")
                )
            else:
                self._docstring_return_type = "-"
