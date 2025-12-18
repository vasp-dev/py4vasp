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
        self._returns_field_type = None
        self._returns_field_description = None
        self._returns_field_body_content = []
        self._in_returns_field = False
        self._visited_definition_list_in_returns = False

    def __str__(self):
        return "\n".join(self.lines) + "\n"

    def find_return_type(self, node) -> tuple[str, str, str]:
        """Retrieves return type information from a desc_signature node.

        Extracts return type from signature annotation and separates Returns field
        type from description.

        Parameters
        ----------
        node : docutils.nodes.Node
            The desc_signature node to retrieve the return type from.

        Returns
        -------
        tuple[str, str, str]
            A tuple of (signature_return_type, returns_field_type, returns_field_description).
            signature_return_type: Type from function signature annotation (empty string if none)
            returns_field_type: Type extracted from Returns field (empty string if none)
            returns_field_description: Description text from Returns field (empty string if none)
        """
        if node.__class__.__name__ != "desc_signature":
            raise UserWarning(
                "Node passed to find_return_type is not a desc_signature node. Return type not retrieved."
            )
        else:
            self._signature_return_type = ""
            self._returns_field_type = ""
            self._returns_field_description = ""
            self._returns_field_body_content = []
            node.parent.walkabout(self)
            # Process collected Returns field body content
            self._parse_returns_field_body()
        return self._signature_return_type, self._returns_field_type, self._returns_field_description

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
                            if type_text and not self._returns_field_type:
                                # Validate that it looks like a type annotation
                                if self._is_type_like(type_text):
                                    self._returns_field_type = type_text
                                else:
                                    # If it's not a type, it's probably a description that Napoleon misplaced
                                    if not self._returns_field_description:
                                        self._returns_field_description = type_text
                elif "return" in field_name and field_name != "return type":
                    self._in_returns_field = True
                    # Capture the field_body content for later parsing
                    for field_child in node.children:
                        if field_child.__class__.__name__ == "field_body":
                            self._capture_field_body_content(field_child)
                break
        pass
    
    def _capture_field_body_content(self, field_body_node):
        """Capture the structure and content of the Returns field body."""
        import sys
        for child in field_body_node.children:
            child_type = child.__class__.__name__
            if child_type == "paragraph":
                # Single or multi-line paragraph
                text = child.astext().strip()
                self._returns_field_body_content.append({
                    'type': 'paragraph',
                    'text': text,
                    'lines': text.split('\n')
                })
            elif child_type == "definition_list":
                # NumPy-style with type and description
                self._visited_definition_list_in_returns = True
                for item in child.children:
                    if item.__class__.__name__ == "definition_list_item":
                        term = None
                        definition = None
                        for item_child in item.children:
                            if item_child.__class__.__name__ == "term":
                                term = item_child.astext().strip()
                            elif item_child.__class__.__name__ == "definition":
                                definition = item_child.astext().strip()
                        if term:
                            self._returns_field_body_content.append({
                                'type': 'definition_list_item',
                                'term': term,
                                'definition': definition or ''
                            })

    def depart_field(self, node):
        # Reset flag when leaving a Returns field
        for child in node.children:
            if child.__class__.__name__ == "field_name":
                field_name = child.astext().strip().lower()
                if "return" in field_name and field_name != "return type":
                    self._in_returns_field = False
                    self._visited_definition_list_in_returns = False
                break

    def visit_desc_returns(self, node):
        if not (self._signature_return_type):
            raw_type = (
                _extract_type_text(node)
                .lstrip(" -> ")
                .strip()
            )
            # Only replace " or " with " | " if it's not inside brackets
            # This prevents Union[int, str] or Tuple[int, str] from being mangled
            import re

            # Check if there are brackets in the type
            if '[' in raw_type and ']' in raw_type:
                # Don't replace " or " inside brackets
                self._signature_return_type = raw_type
            else:
                # Safe to replace " or " with " | " for simple union types
                self._signature_return_type = (
                    raw_type
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
        # Reset flags when leaving any section
        self._in_returns_field = False
        self._visited_definition_list_in_returns = False

    def visit_definition_list(self, node):
        # NumPy-style Returns sections are converted to definition lists
        if self._in_returns_field:
            self._visited_definition_list_in_returns = True
        pass

    def visit_definition_list_item(self, node):
        # Content is captured in _capture_field_body_content
        pass

    def _parse_returns_field_body(self):
        """Parse Returns field body to separate type from description."""
        if not self._returns_field_body_content:
            return
        
        for item in self._returns_field_body_content:
            if item['type'] == 'definition_list_item':
                # NumPy-style: term is type, definition is description
                term = item['term'].strip().strip('`')
                if self._is_type_like(term):
                    self._returns_field_type = term
                    self._returns_field_description = item['definition']
                else:
                    # Term doesn't look like a type, treat as description
                    desc = f"{term}\n{item['definition']}" if item['definition'] else term
                    self._returns_field_description = desc
            elif item['type'] == 'paragraph':
                # Check if it's a single line that looks like a type
                lines = item['lines']
                if len(lines) == 1:
                    # Single line - check if it's a type
                    line = lines[0].strip()
                    if self._is_type_like(line):
                        self._returns_field_type = line
                    else:
                        # Single line description
                        self._returns_field_description = line
                else:
                    # Multiple lines - check if first line is type (NumPy style without definition_list)
                    # This happens when Napoleon processes the docstring
                    first_line = lines[0].strip()
                    # Check indentation of second line relative to first
                    if len(lines) > 1:
                        # If all lines after the first are indented, first line might be type
                        rest_lines = lines[1:]
                        # Check if subsequent lines look like continuation (indented or empty)
                        if all(line.startswith(' ') or not line.strip() for line in rest_lines):
                            # First line is likely a type
                            if self._is_type_like(first_line):
                                self._returns_field_type = first_line
                                desc = '\n'.join(rest_lines).strip()
                                self._returns_field_description = desc
                            else:
                                # Even with indentation, first line doesn't look like a type
                                self._returns_field_description = item['text']
                        else:
                            # Multiple non-indented lines = all description
                            self._returns_field_description = item['text']
                    else:
                        self._returns_field_description = item['text']

    def _is_type_like(self, text: str) -> bool:
        """Check if text looks like a type annotation."""
        import re
        
        text = text.strip().strip('`')
        if not text:
            return False

        import sys
        print("DEBUG: " + text, flush=True, file=sys.stderr)
        
        # Single word (simple type)
        if len(text.split()) == 1 and text.replace('_', '').replace('.', '').replace('[', '').replace(']', '').replace(',', '').isalnum():
            return True
        
        # Generic type like Type[...], Union[...], Tuple[...], etc.
        if re.match(r'^[A-Za-z_][A-Za-z0-9_]*\[.+\]$', text):
            return True
        
        # Union with | or "or"
        if " | " in text or " or " in text:
            parts = text.replace(" or ", " | ").split(" | ")
            if all(
                len(part.strip().split()) == 1 or 
                re.match(r'^[A-Za-z_][A-Za-z0-9_]*\[.+\]$', part.strip())
                for part in parts
            ):
                return True
        
        return False

    def _set_docstring_return_type_if_applicable(self, term_text: str):
        """Deprecated - kept for compatibility but replaced by _is_type_like and _parse_returns_field_body."""
        pass
