# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from docutils.nodes import NodeVisitor, SkipNode


class HugoTranslator(NodeVisitor):
    """A docutils NodeVisitor that translates Sphinx document trees to Hugo-compatible Markdown.

    This translator implements the visitor pattern to traverse Sphinx document trees and convert
    them to Markdown format with Hugo front matter. The visitor pattern is used because it allows
    clean separation of the tree traversal logic from the conversion logic.

    Visit methods (visit_*) are called when entering a node during tree traversal. They are needed
    for every node type that requires special handling when first encountered. Use visit methods to:
    - Add opening markup (like opening tags or markdown syntax)
    - Initialize state for the current node
    - Add content that should appear before the node's children

    Depart methods (depart_*) are called when leaving a node after all its children have been
    processed. Add a depart method when you need to:
    - Add closing markup (like closing tags)
    - Clean up state after processing a node and its children
    - Add content that should appear after the node's children
    - Handle hierarchical structures where you need to track nesting levels

    Many simple nodes only need visit methods since Markdown doesn't require explicit closing
    for most constructs. The unknown_departure method handles cases where no specific depart
    method is defined.
    """

    def __init__(self, document):
        """Initialize the translator with tracking state for Hugo conversion.

        We track has_title to ensure Hugo front matter is only added once at the document start.
        Section level tracking is necessary because Markdown headers are flat (# ## ###) while
        docutils sections are hierarchical, so we need to convert the tree structure to the
        appropriate number of hash symbols.
        """
        self.document = document
        self.has_title = False
        self.section_level = 0
        self.content = []
        self.list_stack = []

    def unknown_visit(self, node):
        """Handle unknown node types by logging them for debugging."""
        # print(f"DEBUG: Unknown node type: {node.__class__.__name__}")
        # print(f"DEBUG: Node attributes: {node.attributes}")
        # print(f"DEBUG: Node children: {[child.__class__.__name__ for child in node.children]}")
        # Don't raise error, just skip for now
        raise NotImplementedError(
            f"Unknown node type {node.__class__.__name__} encountered.\nNode attributes: {node.attributes}\nNode children: {[child.__class__.__name__ for child in node.children]}")

    def unknown_departure(self, node):
        """Handle departure from nodes that don't have specific depart methods.

        Most Markdown constructs don't require closing syntax, so we provide this
        no-op handler to avoid errors when the visitor tries to call depart methods
        that don't exist.
        """
        pass

    def visit_document(self, node):
        """Handle document root node.

        No action needed because Hugo front matter is handled by the first title,
        and we don't want to add any wrapper content around the entire document.
        """
        pass

    def visit_title(self, node):
        """Handle title nodes by generating Hugo front matter and Markdown headers.

        The first title becomes Hugo front matter (TOML format) because Hugo expects
        the page title in front matter rather than as a Markdown header. Subsequent
        titles become Markdown headers with the appropriate nesting level based on
        the current section depth.
        """
        if not self.has_title:
            self.content.append(self._create_hugo_front_matter(node))
            self.has_title = True
        self.content.append(f"{self.section_level * '#'} ")

    def depart_title(self, node):
        self.content.append("\n")

    def _create_hugo_front_matter(self, node):
        """Create Hugo front matter for the document title.
        This method generates the TOML front matter required by Hugo, which includes
        the document title. It is called only once, when the first title node is visited.
        """
        return f"""\
+++
title = "{node.astext()}"
+++\n"""

    def visit_section(self, node):
        """Increment section nesting level when entering a section.

        Sections in docutils are hierarchical containers, but Markdown headers are flat.
        We track the nesting level so that titles within deeper sections get more
        hash symbols (# vs ## vs ###) to maintain the document hierarchy.
        """
        self.section_level += 1

    def depart_section(self, node):
        """Decrement section nesting level when leaving a section.

        This depart method is necessary because we need to restore the previous
        nesting level after processing all content within a section. Without this,
        subsequent sections at the same level would get incorrect header depths.
        """
        self.section_level -= 1

    def visit_paragraph(self, node):
        pass

    def depart_paragraph(self, node):
        self.content.append("\n")

    def visit_Text(self, node):
        self.content.append(f"{node.astext()}")

    # Inline markup handling methods

    def visit_emphasis(self, node):
        self.content.append("*")

    def depart_emphasis(self, node):
        self.content.append("*")

    def visit_strong(self, node):
        self.content.append("**")

    def depart_strong(self, node):
        self.content.append("**")

    def visit_literal(self, node):
        self.content.append("`")

    def depart_literal(self, node):
        self.content.append("`")

    # list handling methods

    def visit_bullet_list(self, node):
        self.list_stack.append("*")

    def depart_bullet_list(self, node):
        self.depart_list()

    def visit_enumerated_list(self, node):
        self.list_stack.append("1.")

    def depart_enumerated_list(self, node):
        self.depart_list()

    def depart_list(self):
        self.list_stack.pop()
        if not self.list_stack:
            self.content.append("\n")  # Add newline after the last list item

    def visit_list_item(self, node):
        indent = "  " * (len(self.list_stack) - 1)
        self.content.append(f"{indent}{self.list_stack[-1]} ")

    def visit_definition_list(self, node):
        pass

    def visit_definition_list_item(self, node):
        pass

    def visit_term(self, node):
        pass

    def depart_term(self, node):
        self.content.append("\n")

    def visit_definition(self, node):
        self.content.append(": ")

    def depart_definition(self, node):
        self.content.append("\n")

    def visit_compound(self, node):
        pass

    def depart_compound(self, node):
        pass

    def visit_comment(self, node):
        raise SkipNode

    def depart_comment(self, node):
        pass

    def visit_compact_paragraph(self, node):
        pass

    def depart_compact_paragraph(self, node):
        self.content.append("\n")

    def visit_reference(self, node):
        # Handle both external and internal references
        self._reference_uri = node.get('refuri') or (f"#{node.get('refid')}" if node.get('refid') else '')
        self.content.append("[")

    def depart_reference(self, node):
        uri = getattr(self, '_reference_uri', '')
        self.content.append(f"]({uri})")
        self._reference_uri = None

    def visit_target(self, node):
        # Internal targets have 'refid'; external have 'refuri'
        refid = node.get('refid')
        if refid:
            # Insert an anchor for internal references
            self.content.append(f'<a name="{refid}"></a>')
        # For external targets (refuri), do nothing (handled by reference)

    def depart_target(self, node):
        pass  # No action needed on depart

    def visit_inline(self, node):
        # Check if this is a cross-reference (std-ref)
        classes = node.get('classes', [])
        if 'std-ref' in classes:
            # This is an internal cross-reference, create a link
            # We need to get the target from somewhere - let's use the text content
            # as the anchor since that's what Sphinx typically does
            self.content.append("[")
            self._inline_is_xref = True
        else:
            self._inline_is_xref = False

    def depart_inline(self, node):
        if getattr(self, '_inline_is_xref', False):
            # Extract the text content to use as the anchor
            text_content = node.astext()
            self.content.append(f"](#{text_content})")
            self._inline_is_xref = False

    def visit_pending_xref(self, node):
        # Use reftarget for anchor
        self._reference_uri = f"#{node.get('reftarget', '')}"
        self.content.append("[")

    def depart_pending_xref(self, node):
        uri = getattr(self, '_reference_uri', '')
        self.content.append(f"]({uri})")
        self._reference_uri = None
