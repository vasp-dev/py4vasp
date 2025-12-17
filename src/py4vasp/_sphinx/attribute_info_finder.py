# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from docutils.nodes import NodeVisitor, SkipNode
from sphinx import addnodes


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


class AttributeInfoFinder(NodeVisitor):
    """A docutils NodeVisitor that finds attribute type and default value information."""

    def __init__(self, document):
        super().__init__(document)
        self._type_annotation = None
        self._default_value = None
        self._in_type_field = False
        self._sig_text = None

    def find_attribute_info(self, node) -> tuple[str | None, str | None]:
        """Retrieve type annotation and default value from a desc_signature node.

        Parameters
        ----------
        node : docutils.nodes.Node
            The desc_signature node to retrieve attribute info from.

        Returns
        -------
        tuple[str | None, str | None]
            A tuple of (type_annotation, default_value). Either may be None if not found.
        """
        if node.__class__.__name__ != "desc_signature":
            raise UserWarning(
                "Node passed to find_attribute_info is not a desc_signature node."
            )

        self._type_annotation = None
        self._default_value = None
        self._sig_text = node.astext()

        # Walk the node tree to find type and default
        node.walkabout(self)

        # If we didn't find them in the tree, try regex parsing
        if not self._type_annotation or not self._default_value:
            self._parse_signature_text()

        return self._type_annotation, self._default_value

    def _parse_signature_text(self):
        """Parse signature text using regex as fallback."""
        import re

        if not self._sig_text:
            return

        # Remove attribute/property keywords
        clean_sig = re.sub(r"^(attribute|property)\s+", "", self._sig_text)

        # Look for ": type" (type is everything between : and = or end, excluding =)
        if not self._type_annotation:
            type_match = re.search(r":\s*([^=]+?)(?:\s*=|$)", clean_sig)
            if type_match:
                potential_type = type_match.group(1).strip()
                # Make sure it's not empty and doesn't start with =
                if potential_type and not potential_type.startswith("="):
                    self._type_annotation = potential_type

        # Look for "= value"
        if not self._default_value:
            default_match = re.search(r"=\s*(.+)$", clean_sig)
            if default_match:
                self._default_value = default_match.group(1).strip()

    def unknown_visit(self, node):
        """Skip unknown node types by default."""
        raise SkipNode

    def unknown_departure(self, node):
        """Handle departure from nodes that don't have specific depart methods."""
        pass

    def visit_desc_signature(self, node):
        """Visit the signature node to extract information."""
        pass

    def visit_desc_type(self, node):
        """Extract type from desc_type node."""
        if not self._type_annotation:
            text = _extract_type_text(node).strip()
            # Skip annotation markers and assignment operators
            if (
                text
                and text not in ("property", "attribute", ":", "=", "")
                and not text.startswith("=")
            ):
                self._type_annotation = text.lstrip(":").strip()
        raise SkipNode

    def visit_desc_annotation(self, node):
        """Extract default value from desc_annotation node."""
        text = node.astext().strip()
        # Skip annotation markers and assignment operators
        if (
            text
            and text not in ("property", "attribute", ":", "=", "")
            and not text.startswith("=")
        ):
            # Check if this is a default value (starts with =)
            if text.startswith("=") and not self._default_value:
                self._default_value = text[1:].strip()
            # Check if this is a type annotation
            elif not text.startswith("=") and not self._type_annotation:
                self._type_annotation = text.lstrip(":").strip()
        raise SkipNode

    def visit_pending_xref(self, node):
        """Extract type from pending cross-reference nodes."""
        if not self._type_annotation:
            text = _extract_type_text(node).strip()
            if (
                text
                and text not in ("property", "attribute", ":", "=", "")
                and not text.startswith("=")
            ):
                self._type_annotation = text
        raise SkipNode

    def visit_desc_sig_literal_number(self, node):
        """Extract numeric default value."""
        if not self._default_value and "=" in self._sig_text:
            val = node.astext().strip()
            if val and val != "=":
                self._default_value = val
        raise SkipNode

    def visit_desc_sig_literal_string(self, node):
        """Extract string default value."""
        if not self._default_value and "=" in self._sig_text:
            val = node.astext().strip()
            if val and val != "=":
                self._default_value = val
        raise SkipNode

    def visit_inline(self, node):
        """Extract default value from inline nodes."""
        if not self._default_value and "=" in self._sig_text:
            val = node.astext().strip()
            if val and val != "=":
                self._default_value = val
        raise SkipNode

    def visit_field_list(self, node):
        """Look for type information in field lists."""
        pass

    def visit_field(self, node):
        """Check if this is a type field."""
        pass

    def visit_field_name(self, node):
        """Check field name for 'type'."""
        if "type" in node.astext().lower():
            self._in_type_field = True
        raise SkipNode

    def visit_field_body(self, node):
        """Extract type from field body if in type field."""
        if self._in_type_field and not self._type_annotation:
            self._type_annotation = _extract_type_text(node).strip()
        raise SkipNode

    def depart_field(self, node):
        """Reset type field flag."""
        self._in_type_field = False

    def visit_desc_content(self, node):
        """Visit desc_content to look for field lists."""
        pass
