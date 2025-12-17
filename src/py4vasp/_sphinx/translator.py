# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pathlib
from datetime import datetime
from typing import Optional

from docutils.nodes import NodeVisitor, SkipNode

from py4vasp import _calculation
from py4vasp._sphinx.anchors_finder import AnchorsFinder
from py4vasp._sphinx.attribute_info_finder import AttributeInfoFinder
from py4vasp._sphinx.parameters_info_finder import (
    SIG_TYPE_DEFAULT,
    ParametersInfoFinder,
    _get_param_raw_info_from_left_string,
)
from py4vasp._sphinx.return_type_finder import ReturnTypeFinder


def _construct_hugo_shortcode(text: str) -> str:
    """Wrap text in Hugo shortcode delimiters."""
    return f"{{{{< {text} >}}}}"


class Indentation:
    """A simple data class to track the current indentation level."""

    def __init__(self, level: int, level_in_first_row: Optional[int] = None):
        self._level = level
        self._level_in_first_row = level_in_first_row
        self._row = 0

    @property
    def level(self):
        if self._row > 0 or self._level_in_first_row is None:
            return self._level
        else:
            return self._level_in_first_row

    def indent(self, line: str) -> str:
        """Indent a line based on the current indentation level.

        If level_in_first_row is set, it uses that for the first row,
        otherwise it uses the current level."""
        result = f"{'    ' * self.level}{line}"
        self._row += 1
        return result


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
        # print("DEBUG:", document.pformat())
        self.document = document
        self.frontmatter_created = False
        self.section_level = 0
        self.indentation_stack = [Indentation(0)]
        self.lines = []
        self.content = ""
        self.list_stack = []
        self.anchor_id_stack = []
        """Used to identify stacks of anchor IDs for desc_signature, so that class methods, e.g. can be referenced as #script_name.Class_name.method_name."""
        self._in_parameters_field = False
        self._in_returns_field = False
        self._in_examples_field = False
        self._expect_returns_field = False
        self._is_shortcode_docstring_open = False
        self._is_shortcode_sphinx_open = False
        self._needs_reopen_docstring = False
        self._current_return_type = None
        self._current_signature_dict = {}
        self._prevent_move_content = False
        self._prevent_content_stash_deletion = False
        self._content_stash = []

    def __str__(self):
        return "\n".join(self.lines) + "\n"

    def _shortcode_sphinx(self, close: bool = False):
        if close:
            self._shortcode_docstring(close=True)
            if self._is_shortcode_sphinx_open:
                self.content += f"\n\n{_construct_hugo_shortcode('/sphinx')}\n"
                self._is_shortcode_sphinx_open = False
        else:
            if not self._is_shortcode_sphinx_open:
                self.content += f"\n{_construct_hugo_shortcode('sphinx')}\n\n"
                self._is_shortcode_sphinx_open = True

    def _shortcode_docstring(self, close: bool = False):
        if close:
            if self._is_shortcode_docstring_open:
                self.content += f"{_construct_hugo_shortcode('/docstring')}"
                self._is_shortcode_docstring_open = False
        else:
            if not self._is_shortcode_docstring_open:
                self.content += f"{_construct_hugo_shortcode('docstring')}\n"
                self._is_shortcode_docstring_open = True

    def _add_new_line(self):
        if self._prevent_move_content:
            self._content_stash.append("")
        else:
            self.lines.append("")

    def _strip_blank_lines(self):
        while self.lines and not self.lines[-1]:
            self.lines.pop()

    def _move_content_to_lines(self):
        if not (self._prevent_move_content) and self._content_stash:
            for line in self._content_stash:
                self.lines.append(line)
            if not (self._prevent_content_stash_deletion):
                self._content_stash = []

        for line in self.content.splitlines():
            full_line = self.indentation_stack[-1].indent(line).rstrip()
            if not self._prevent_move_content:
                self.lines.append(full_line)
            else:
                self._content_stash.append(full_line)
        self.content = ""

    def unknown_visit(self, node):
        """Handle unknown node types by logging them for debugging."""
        # print(f"DEBUG: Unknown node type: {node.__class__.__name__}")
        # print(f"DEBUG: Node attributes: {node.attributes}")
        # print(
        #     f"DEBUG: Node children: {[child.__class__.__name__ for child in node.children]}"
        # )
        # Don't raise error, just skip for now
        return
        raise NotImplementedError(
            f"Unknown node type {node.__class__.__name__} encountered.\nNode attributes: {node.attributes}\nNode children: {[child.__class__.__name__ for child in node.children]}"
        )

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

    def depart_document(self, node):
        """Close all open shortcodes and finalize document."""
        self._shortcode_docstring(close=True)
        self._shortcode_sphinx(close=True)
        self._move_content_to_lines()

    def visit_title(self, node):
        """Handle title nodes by generating Hugo front matter and Markdown headers.

        The first title becomes Hugo front matter (TOML format) because Hugo expects
        the page title in front matter rather than as a Markdown header. Subsequent
        titles become Markdown headers with the appropriate nesting level based on
        the current section depth.
        """
        self._create_hugo_front_matter(node)
        self._shortcode_sphinx()
        self._shortcode_docstring()
        self._move_content_to_lines()
        if self.section_level > 1:
            self.content += f"{self.section_level * '#'} "
        else:
            raise SkipNode

    def depart_title(self, node):
        self._move_content_to_lines()

    def _create_hugo_front_matter(self, node):
        """Create Hugo front matter for the document title.
        This method generates the TOML front matter required by Hugo, which includes
        the document title. It is called only once, when the first title node is visited.
        """
        if self.frontmatter_created:
            return
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.content += f"""\
+++
title = "{node.astext()}"
weight = HUGO_WEIGHT_PLACEHOLDER
date = "{current_date}"
+++\n"""
        self._move_content_to_lines()
        self.frontmatter_created = True

    def visit_section(self, node):
        """Increment section nesting level when entering a section.

        Sections in docutils are hierarchical containers, but Markdown headers are flat.
        We track the nesting level so that titles within deeper sections get more
        hash symbols (# vs ## vs ###) to maintain the document hierarchy.
        """
        self.section_level += 1
        # Open docstring for content sections (between desc nodes)
        if self._is_shortcode_sphinx_open:
            self._shortcode_docstring(close=False)

    def depart_section(self, node):
        """Decrement section nesting level when leaving a section.

        This depart method is necessary because we need to restore the previous
        nesting level after processing all content within a section. Without this,
        subsequent sections at the same level would get incorrect header depths.
        """
        # Close docstring for content sections, but not if we need to reopen it
        # (e.g., after a compound or admonition)
        if self._is_shortcode_docstring_open and not self._needs_reopen_docstring:
            self._shortcode_docstring(close=True)
        self.section_level -= 1

    def visit_paragraph(self, node):
        """Empty visit method because paragraph opening requires no markup.

        Unlike HTML, Markdown paragraphs don't need opening tags or special markers.
        The content will be added by child Text nodes, and spacing is handled in depart.
        Inside lists only a single newline is required because the list item separate
        the paragraphs.
        """
        pass

    def depart_paragraph(self, node):
        """Add newline after paragraph content for proper Markdown separation.

        Markdown requires blank lines between block elements. We handle this in depart
        rather than visit because we need the newline after all the paragraph's content
        has been processed, not before it.
        """
        self._move_content_to_lines()
        self._add_new_line()

    def visit_rubric(self, node):
        self.content += "\n" + (self.section_level + 1) * "#" + " "

    def depart_rubric(self, node):
        self.content += "\n\n"
        self._move_content_to_lines()

    def visit_Text(self, node):
        """Add text content directly without modification.

        Text nodes contain the raw content and don't need escaping or wrapping
        in basic cases. More complex escaping could be added here if needed.
        """
        self.content += node.astext()

    # Inline markup handling methods

    def visit_emphasis(self, node):
        """Add opening asterisk for italic text.

        Markdown uses single asterisks for emphasis. We use asterisks instead of
        underscores because they're more universally supported and don't conflict
        with underscores in code or identifiers.
        """
        self.content += "*"

    def depart_emphasis(self, node):
        """Add closing asterisk to complete italic markup.

        Both opening and closing markers are identical in Markdown emphasis,
        unlike HTML where tags differ (&lt;em&gt; vs &lt;/em&gt;).
        """
        self.content += "*"

    def visit_strong(self, node):
        """Add opening double asterisk for bold text.

        Markdown uses double asterisks for strong/bold text. Double asterisks are
        preferred over double underscores for consistency with single emphasis.
        """
        self.content += "**"

    def depart_strong(self, node):
        """Add closing double asterisk to complete bold markup.

        Symmetric opening and closing markers are required for proper Markdown parsing.
        """
        self.content += "**"

    def visit_math(self, node):
        """Handle inline math nodes by converting to KaTeX format.

        Sphinx :math: role creates math nodes. We convert these to $...$
        for KaTeX/MathJax rendering in Hugo.
        """
        self.content += "$"

    def depart_math(self, node):
        """Close inline math with dollar sign."""
        self.content += "$"

    def visit_literal(self, node):
        """Add opening backtick for inline code.

        Single backticks are used for inline code spans in Markdown. This handles
        simple cases where the literal text doesn't contain backticks itself.
        Skip backticks if the literal contains a pending cross-reference (xref),
        since those should be rendered as links, not code.
        """
        # Check if this literal contains a pending_xref (for :meth:, :class:, etc.)
        has_xref = any(
            child.__class__.__name__ == "pending_xref" for child in node.children
        )
        if not has_xref:
            self.content += "`"
        self._literal_has_xref = has_xref

    def depart_literal(self, node):
        """Add closing backtick to complete inline code markup.

        Matching backticks are required to properly delimit the code span.
        Skip if this literal contained a cross-reference.
        """
        if getattr(self, "_table_style", "") == "autosummary":
            for quantity_name, alias in _calculation.GROUP_TYPE_ALIAS.items():
                self.content = self.content.replace(quantity_name, alias)
        if not getattr(self, "_literal_has_xref", False):
            self.content += "`"
        self._literal_has_xref = False

    # list handling methods

    def visit_bullet_list(self, node):
        """Initialize bullet list state by pushing marker onto stack.

        We use a stack to handle nested lists properly. Each nesting level needs
        to know what marker to use, and asterisks are chosen over dashes for
        better Hugo/CommonMark compatibility.
        """
        self.list_stack.append("*")

    def depart_bullet_list(self, node):
        """Clean up bullet list state when exiting the list.

        Uses the shared depart_list method because the cleanup logic is identical
        for both bullet and enumerated lists - only the markers differ.
        """
        self._depart_list()

    def visit_enumerated_list(self, node):
        """Initialize enumerated list state with numbered marker.

        We use "1." for all items rather than tracking actual numbers because
        Markdown auto-numbers list items, making the specific number irrelevant.
        This simplifies the implementation without affecting output.
        """
        self.list_stack.append("1.")

    def depart_enumerated_list(self, node):
        """Clean up enumerated list state when exiting the list.

        Shares cleanup logic with bullet lists since the behavior is identical
        regardless of marker type.
        """
        self._depart_list()

    def _depart_list(self):
        """Remove current list from stack and add spacing after top-level lists.

        We only add newlines after the outermost list (when stack becomes empty)
        because nested lists shouldn't have extra spacing between them, but
        lists need separation from following content.
        """
        self.list_stack.pop()
        if not self.list_stack:
            self._add_new_line()

    def visit_list_item(self, node):
        """Add list marker with proper indentation for current nesting level.

        Indentation is calculated from stack depth to handle nested lists correctly.
        Two spaces per level is the Markdown standard for nested list indentation.
        The marker comes from the stack top, so it matches the current list type.
        """
        indentation = Indentation(
            level=len(self.list_stack), level_in_first_row=len(self.list_stack) - 1
        )
        self.indentation_stack.append(indentation)
        self.content += f"{self.list_stack[-1]:4}"

    def depart_list_item(self, node):
        self.indentation_stack.pop()

    def visit_definition_list(self, node):
        self.list_stack.append("description")

    def depart_definition_list(self, node):
        self._depart_list()

    def visit_definition_list_item(self, node):
        """No action needed for definition list item container.

        Like the parent definition list, individual items don't need wrapper markup.
        The term and definition children handle their own formatting.
        """
        pass

    def visit_term(self, node):
        """No opening markup needed for definition terms.

        In Markdown definition lists, terms are just regular text that will be
        followed by a colon and definition. No special markup is required.
        """
        pass

    def depart_term(self, node):
        """Add newline after term to separate it from the definition.

        This creates the line break that Markdown definition list syntax requires
        between the term and the definition that follows.
        """
        self._move_content_to_lines()

    def visit_definition(self, node):
        """Add colon prefix to mark the beginning of a definition.

        This follows Markdown definition list syntax where definitions are
        prefixed with a colon and space. The colon signals that this is the
        definition part of the term-definition pair.
        """
        self.content += ": <!---->"
        self._move_content_to_lines()
        indentation = Indentation(self.indentation_stack[-1].level + 1)
        self.indentation_stack.append(indentation)

    def depart_definition(self, node):
        """Add newline after definition for proper separation.

        This ensures proper spacing between definition list items and prevents
        them from running together in the output.
        """
        self._move_content_to_lines()
        self.indentation_stack.pop()

    # Comment handling methods

    def visit_comment(self, node):
        """Skip comment nodes entirely to exclude them from output.

        Comments are typically internal documentation that shouldn't appear
        in the final output. SkipNode prevents the visitor from processing
        this node's children, effectively removing it from the output.
        """
        raise SkipNode

    def depart_comment(self, node):
        """No action needed since comments are skipped entirely.

        This method exists only to satisfy the visitor pattern expectations,
        but will never be called due to SkipNode being raised in visit.
        """
        pass

    # Internal and external reference handling methods

    def visit_reference(self, node):
        """Start a Markdown link and store the target URI for later use.

        We handle both external URLs (refuri) and internal anchors (refid) with
        the same link syntax. Internal references get a hash prefix to create
        anchor links. The URI is stored as an instance variable because we need
        it in the depart method after processing the link text.
        """
        self.content += "["

    def depart_reference(self, node):
        """Complete the Markdown link with the stored URI and clean up state.

        The link text has been added by child nodes between visit and depart,
        so we can now close the link with the URI that was stored in visit.
        We clean up the temporary state to avoid interference with other links.
        """
        reference = node.get("reftitle", "")
        anchor = ""
        if not reference:
            link = node.get("refuri", "")
            self.content += f"]({link})"
            return
        if reference == "py4vasp.Calculation":
            reference = "calculation/calculation"
        elif reference.startswith("py4vasp.Calculation"):
            parts = reference.split(".")
            assert len(parts) == 3, f"Invalid reference format: {reference}"
            if parts[2] in _calculation.QUANTITIES:
                reference = f"calculation/{parts[2]}"
            else:
                reference = f"calculation/calculation"
                anchor = f"Calculation-{parts[2]}"
        elif reference.startswith("py4vasp._calculation"):
            parts = reference.split(".")
            assert len(parts) >= 3, f"Invalid reference format: {reference}"
            reference = f"calculation/{parts[2]}"
            if len(parts) >= 5:
                anchor = f"{parts[3]}-{parts[4]}"
        else:
            reference = reference.removeprefix("py4vasp.")
            reference = reference.replace(".", "/")
        reference = pathlib.Path(reference)
        # index files are represented by folder names in Hugo
        source = self.document.source
        source = source.removesuffix("_index").removesuffix("index")
        source = pathlib.Path(source)
        relative_path = reference.relative_to(source, walk_up=True).as_posix()
        if anchor:
            self.content += f"]({relative_path}#{anchor})"
        else:
            self.content += f"]({relative_path})"

    def visit_target(self, node):
        """Create HTML anchor tags for internal reference targets.

        Targets with refid become anchor points that can be linked to from elsewhere
        in the document. We use HTML anchor tags because Markdown doesn't have
        native syntax for arbitrary anchor points. External targets (refuri) are
        handled by the reference nodes that point to them.
        """
        # Internal targets have 'refid'; external have 'refuri'
        refid = node.get("refid")
        if refid:
            docstring_cont = False
            if self._is_shortcode_docstring_open:
                docstring_cont = True
                self._shortcode_docstring(close=True)
            # Insert an anchor for internal references
            self.content += _construct_hugo_shortcode(f'anchor name="{refid}"')
            self.content += _construct_hugo_shortcode(f"/anchor")
            if docstring_cont:
                self._shortcode_docstring()
        # For external targets (refuri), do nothing (handled by reference)

    def depart_target(self, node):
        """No cleanup needed for target nodes.

        Anchor tags are self-contained and don't require closing markup or
        state cleanup like some other node types.
        """
        pass

    def visit_inline(self, node):
        """Handle inline nodes, particularly Sphinx cross-references.

        Sphinx uses inline nodes with specific CSS classes to mark cross-references.
        We detect 'std-ref' class to identify internal cross-references and treat
        them as links. Other inline nodes are ignored since they typically don't
        need special Markdown markup.
        """
        # Check if this is a cross-reference (std-ref)
        classes = node.get("classes", [])
        if "std-ref" in classes:
            # This is an internal cross-reference, create a link
            # We need to get the target from somewhere - let's use the text content
            # as the anchor since that's what Sphinx typically does
            self.content += "["
            self._inline_is_xref = True
        else:
            self._inline_is_xref = False

    def depart_inline(self, node):
        """Complete cross-reference links using the node's text as the anchor.

        For cross-references, we use the displayed text as the anchor target,
        which is a common convention in documentation. This assumes the text
        content matches an existing anchor somewhere in the document.
        """
        if getattr(self, "_inline_is_xref", False):
            # Extract the text content to use as the anchor
            text_content = node.astext()
            self.content += f"](#{text_content})"
            self._inline_is_xref = False

    def visit_pending_xref(self, node):
        """Handle Sphinx pending cross-references by creating anchor links.

        Pending cross-references are unresolved references that Sphinx will
        process later. We extract the target from the 'reftarget' attribute
        and create a link assuming it will resolve to an internal anchor.
        For method references (:meth:), we'll add () suffix after the link.
        """
        # Store reftype to handle special cases like meth
        self._xref_reftype = node.get("reftype", "")
        reftarget = node.get("reftarget", "")
        # For cross-file references, use the full target as anchor
        self._reference_uri = f"#{reftarget}"
        self.content += "["

    def depart_pending_xref(self, node):
        """Complete pending cross-reference links with the stored target.

        Similar to regular references, we close the link with the URI that was
        stored in visit and clean up the temporary state.
        For method references, add () suffix after the link.
        """
        uri = getattr(self, "_reference_uri", "")
        reftype = getattr(self, "_xref_reftype", "")
        self.content += f"]({uri})"
        # Add () suffix for method references
        if reftype == "meth":
            self.content += "()"
        self._reference_uri = None
        self._xref_reftype = None

    # Code block handling methods

    def visit_literal_block(self, node):
        language = node["language"].strip("default")
        self.content += f"~~~{language}\n"

    def depart_literal_block(self, node):
        self.content += "\n~~~"
        self._move_content_to_lines()
        self._add_new_line()

    def visit_doctest_block(self, node):
        self.content += "~~~python\n"

    def depart_doctest_block(self, node):
        self.content += "\n~~~"
        self._move_content_to_lines()
        self._add_new_line()

    # Admonition handling methods

    def visit_note(self, node):
        self._visit_admonition("info")

    def visit_warning(self, node):
        self._visit_admonition("warning")

    def visit_important(self, node):
        self._visit_admonition("primary")

    def visit_tip(self, node):
        self._visit_admonition("success")

    def visit_caution(self, node):
        self._visit_admonition("danger")

    def _visit_admonition(self, type):
        if self._is_shortcode_docstring_open:
            self._shortcode_docstring(close=True)
            self._needs_reopen_docstring = True
        self.content += f'{{{{< admonition type="{type}" >}}}}\n'
        self._move_content_to_lines()
        self.indentation_stack.append(Indentation(0))

    def depart_note(self, node):
        self._depart_admonition()

    def depart_warning(self, node):
        self._depart_admonition()

    def depart_important(self, node):
        self._depart_admonition()

    def depart_tip(self, node):
        self._depart_admonition()

    def depart_caution(self, node):
        self._depart_admonition()

    def _depart_admonition(self):
        self._strip_blank_lines()
        self.indentation_stack.pop()
        self.content += "{{< /admonition >}}\n"
        if self._needs_reopen_docstring:
            self._shortcode_docstring()
            self._needs_reopen_docstring = False
        self._move_content_to_lines()

    # Footnote handling methods

    def visit_footnote_reference(self, node):
        self.content = self.content.strip() + "[^"

    def depart_footnote_reference(self, node):
        self.content += "]"

    def visit_footnote(self, node):
        pass

    def depart_footnote(self, node):
        self.indentation_stack.pop()

    def visit_label(self, node):
        self.content = "[^"

    def depart_label(self, node):
        self.content += "]:"
        self._move_content_to_lines()
        self.indentation_stack.append(Indentation(1))

    # Compound handling methods

    def visit_compound(self, node):
        # Close docstring before compound, like we do for admonitions
        if self._is_shortcode_docstring_open:
            self._shortcode_docstring(close=True)
            self._needs_reopen_docstring = True
        # Start a div to preserve grouping in Markdown
        self.content += f'\n{_construct_hugo_shortcode("compound")}\n\n'

    def depart_compound(self, node):
        # End the div
        self.content += f"\n{_construct_hugo_shortcode('/compound')}\n"
        # Reopen docstring after compound if needed
        if self._needs_reopen_docstring:
            self._shortcode_docstring()
            # DON'T reset _needs_reopen_docstring - let the parent container handle it
        self._move_content_to_lines()

    def visit_compact_paragraph(self, node):
        pass

    def depart_compact_paragraph(self, node):
        pass

    # AUTODOC handling methods
    # === .. autodata:: ===

    def visit_index(self, node):
        pass

    def depart_index(self, node):
        pass

    def _get_anchor_id(self):
        if not self.anchor_id_stack:
            return None
        else:
            list_to_join = [
                anchor.replace(" ", "")
                for (anchor, name, addname, domain, objtype) in self.anchor_id_stack
                if anchor
            ]
            return ".".join(list_to_join)

    def _get_breadcrumbs(self) -> list[str]:
        """Get the module name from the anchor_id_stack."""
        full_anchor = self._get_anchor_id()
        if full_anchor:
            name = self._get_latest_name()
            if name and full_anchor.endswith(name):
                module = full_anchor[: -len(name)].rstrip(".")
                return module.split(".") if module else []
        return []

    def _construct_new_anchor_id(self, node) -> tuple[str, str, str]:
        anchors_finder = AnchorsFinder(self.document)
        name, addname, objtype, domain = anchors_finder.find_anchor(node)
        # Build anchor id (e.g., py-example or py-example-classname)
        new_anchor_id = f"{addname}.{name}".replace(" ", "-").rstrip("-").strip(".")
        self.anchor_id_stack.append((new_anchor_id, name, addname, domain, objtype))
        return new_anchor_id, domain, objtype

    def _get_latest_objtype(self) -> str | None:
        """Get the latest non-empty objtype from the anchor_id_stack."""
        if not self.anchor_id_stack:
            return None
        for anchor_id, name, addname, domain, objtype in reversed(self.anchor_id_stack):
            if objtype:
                return objtype
        return None

    def _get_latest_name(self) -> str | None:
        """Get the latest non-empty name from the anchor_id_stack."""
        if not self.anchor_id_stack:
            return None
        for anchor_id, name, addname, domain, objtype in reversed(self.anchor_id_stack):
            if name:
                return name
        return None

    def visit_desc(self, node):
        self._construct_new_anchor_id(node)
        self.section_level += 1
        breadcrumbs = self._get_breadcrumbs()
        if breadcrumbs and breadcrumbs[0] == "py4vasp":
            breadcrumbs.pop(0)
        if breadcrumbs and breadcrumbs[0] == "_calculation":
            breadcrumbs[0] = "calculation"
        objtype = self._get_latest_objtype()
        name = self._get_latest_name()

        self._shortcode_docstring(close=True)

        if objtype in ["method", "property", "attribute"]:
            assert breadcrumbs, "breadcrumbs should contain at least the class name"
            class_name = breadcrumbs.pop()
            should_skip = (
                class_name != "Calculation"
                and objtype == "method"
                and name in ("from_path", "from_file", "from_data")
            )
            if should_skip:
                self.section_level -= 1
                raise SkipNode
            module_name = self._get_module_name(breadcrumbs)
            shortcode_str = f"{objtype} name=\"{name}\" class=\"{class_name}\" module=\"{module_name}\" breadcrumbs=\"{'.'.join(breadcrumbs)}\""
        elif objtype in ["class", "function", "data"]:
            module_name = self._get_module_name(breadcrumbs)
            shortcode_str = f"{objtype} name=\"{name}\" module=\"{module_name}\" breadcrumbs=\"{'.'.join(breadcrumbs)}\""
        else:
            raise NotImplementedError(f"Unsupported objtype '{objtype}' in visit_desc.")
        self.content += f"\n\n{_construct_hugo_shortcode(shortcode_str)}"

    def _get_module_name(self, breadcrumbs):
        module_name = breadcrumbs.pop() if breadcrumbs else ""
        for group, members in _calculation.GROUPS.items():
            for member in members:
                if module_name == f"{group}_{member}":
                    return f"{group}.{member}"
        return module_name

    def depart_desc(self, node):
        objtype = self._get_latest_objtype()
        if self.anchor_id_stack:
            self.anchor_id_stack.pop()
        self.section_level -= 1

        self._shortcode_docstring(close=True)

        self.content += f"\n\n{_construct_hugo_shortcode(f'/{objtype}')}\n\n"
        self._move_content_to_lines()

    def _get_parameter_list_and_types(self, node, skip_content: bool = False):
        """Extract parameter names, types, and default values from a desc_signature node."""
        parameters_info_finder = ParametersInfoFinder(self.document)
        parameters_dict = parameters_info_finder.find_parameters_info(
            node, skip_content=skip_content
        )
        parameters = []
        for name, info in parameters_dict.items():
            # Reconstruct name with asterisks from signature
            asterisks = info.get("asterisks", "")
            full_name = asterisks + name
            parameters.append((full_name, info.get("type"), info.get("default")))

        if parameters:
            self._current_signature_dict["sig_parameters"] = parameters.copy()
        return parameters

    def _get_formatted_param(self, name, annotation, default):
        """Format a single parameter with its name, type, and default value.

        The name already includes any unpacking asterisks (*args, **kwargs) from the signature.
        """
        param = f"*{name}*"
        if annotation:
            param += ": "
            formatted_annotation = annotation.replace("` or `", " or ").replace(
                " or ", " | "
            )
            formatted_annotation = f"`{formatted_annotation}`"
            # now make sure Markdown links are formatted correctly
            formatted_annotation = formatted_annotation.replace("`[", "[").replace(
                ")`", ")"
            )
            param += formatted_annotation
        if default:
            formatted_default = (
                f" = {default}" if not (default == SIG_TYPE_DEFAULT) else ""
            )
            param += f"{formatted_default}"
        return param

    def _get_parameter_list_str(self, parameters):
        """Get a string representation of the parameter list with types."""
        if not parameters:
            return "\n()"
        param_strs = []
        for name, annotation, default in parameters:
            param = self._get_formatted_param(name, annotation, default)
            param_strs.append(param)
        if len(param_strs) == 1:
            return f"\n({param_strs[0]})"
        elif len(param_strs) == 2 and param_strs == ["**args*", "***kwargs*"]:
            return "\n(**args*, ***kwargs*)"
        concat_str = ",\n- ".join(param_strs)
        return "\n(\n- " + concat_str + "\n\n)"

    def _get_return_type(self, node):
        """Get the return type annotation from a desc_signature node."""
        return_type_finder = ReturnTypeFinder(self.document)
        return_type = return_type_finder.find_return_type(node)
        if return_type:
            self._current_signature_dict["sig_return_type"] = return_type
            self._current_return_type = return_type
            self._expect_returns_field = True
        return return_type

    def _get_attribute_info(self, node):
        """Get type annotation and default value for an attribute/property using AttributeInfoFinder."""
        attribute_info_finder = AttributeInfoFinder(self.document)
        return attribute_info_finder.find_attribute_info(node)

    def visit_desc_signature(self, node):
        objtype = self._get_latest_objtype()
        self._current_signature_dict = {}
        self._current_return_type = None
        if objtype in [
            "function",
            "method",
            "class",
            "exception",
            "property",
            "attribute",
        ]:
            self.content += "\n" + _construct_hugo_shortcode("signature")

            # For attributes/properties, show type and default in signature
            if objtype in ["property", "attribute"]:
                type_annotation, default_value = self._get_attribute_info(node)
                if type_annotation or default_value:
                    type_str = f": `{type_annotation}`" if type_annotation else ""
                    default_str = f" = {default_value}" if default_value else ""
                    self.content += f"{type_str}{default_str}"
                # Try to get return type for properties
                if objtype == "property":
                    return_type = self._get_return_type(node)
                    if return_type:
                        return_str = f" → `{return_type}`"
                        self.content += return_str
            else:
                # For methods/functions/classes, show parameters normally
                parameters = self._get_parameter_list_and_types(
                    node, not (objtype in ["function", "method"])
                )
                parameters_str = self._get_parameter_list_str(parameters)
                self.content += parameters_str
                if objtype in ["function", "method"]:
                    return_type = self._get_return_type(node)
                    if return_type:
                        return_str = f" → `{return_type}`"
                        self.content += return_str

            self.content += "\n" + _construct_hugo_shortcode("/signature")

        self.content += "\n\n"

        if self._current_return_type:
            self._expect_returns_field = True
        raise SkipNode

    def depart_desc_signature(self, node):
        pass

    def visit_desc_returns(self, node):
        raise SkipNode

    def depart_desc_returns(self, node):
        pass

    def visit_desc_content(self, node):
        self._shortcode_docstring()

    def depart_desc_content(self, node):
        self._move_content_to_lines()

    def visit_desc_addname(self, node):
        raise SkipNode

    def depart_desc_addname(self, node):
        pass

    def visit_desc_name(self, node):
        raise SkipNode

    def depart_desc_name(self, node):
        pass

    def visit_desc_annotation(self, node):
        raise SkipNode

    def depart_desc_annotation(self, node):
        pass

    def visit_desc_sig_space(self, node):
        # Space in signatures (for formatting)
        self.content += " "

    def depart_desc_sig_space(self, node):
        pass

    # === .. autoclass:: ===

    def visit_desc_parameterlist(self, node):
        raise SkipNode

    def depart_desc_parameterlist(self, node):
        pass

    def visit_desc_parameter(self, node):
        raise SkipNode

    def depart_desc_parameter(self, node):
        pass

    def visit_desc_sig_name(self, node):
        pass

    def depart_desc_sig_name(self, node):
        pass

    def visit_desc_sig_punctuation(self, node):
        # Punctuation in signature (e.g., '=')
        pass

    def depart_desc_sig_punctuation(self, node):
        pass

    def visit_desc_sig_space(self, node):
        # Space in signature
        pass

    def depart_desc_sig_space(self, node):
        pass

    def visit_desc_sig_operator(self, node):
        # Operator in signature (e.g., '->', '=', etc.)
        pass

    def depart_desc_sig_operator(self, node):
        pass

    def visit_field_list(self, node):
        # Start field list (e.g., for :param:, :returns:, etc.)
        self.content += "\n"

    def depart_field_list(self, node):
        if getattr(self, "_expect_returns_field", False):
            # If we expected a returns field, but didn't find it, log a warning
            if self._current_return_type:
                self.content += self._get_formatted_field_header("Returns")
                self.content += f"`{self._current_return_type}`\n\n"
            self._expect_returns_field = False
            self._current_return_type = None
        self.content += "\n"

    def _get_formatted_field_header(self, field_name):
        """Format the field name for Markdown headers."""
        return f"\n{(self.section_level + 1) * '#'} {field_name.capitalize()}\n\n"

    def visit_field(self, node):
        # Identify the field name
        field_name = ""
        for child in node.children:
            if child.__class__.__name__ == "field_name":
                field_name = child.astext().strip().lower()
                break

        if field_name == "return type":
            if not (self._current_return_type):
                self._current_return_type = getattr(
                    self._current_signature_dict, "sig_return_type", None
                )
            raise SkipNode

        if field_name == "returns":
            self.content += self._get_formatted_field_header("Returns")
            self._in_returns_field = True
            self._move_content_to_lines()
            return

        if field_name == "parameters":
            self.content += self._get_formatted_field_header("Parameters")
            self._in_parameters_field = True
            self._move_content_to_lines()
            return

        if field_name == "examples":
            self.content += self._get_formatted_field_header("Examples")
            self._move_content_to_lines()
            self._in_examples_field = True
            return

    def depart_field(self, node):
        if getattr(self, "_in_returns_field", False):
            self._in_returns_field = False
            self._expect_returns_field = False
        if getattr(self, "_in_parameters_field", False):
            self._in_parameters_field = False
        if getattr(self, "_in_examples_field", False):
            self._in_examples_field = False

    def visit_field_name(self, node):
        raise SkipNode

    def depart_field_name(self, node):
        pass

    def _restructure_parameters_field_body(self):
        pure_str_content = self._content_stash
        new_str_content = ""
        opened_description_list = False
        description_lines = []  # Collect lines for a single parameter description

        for line in pure_str_content:
            if not (" – " in line):
                if not (opened_description_list):
                    new_str_content += "\n: <!---->"
                    opened_description_list = True
                # Collect description lines to process as a group
                description_lines.append(line)
            else:
                # Process any accumulated description lines first
                if description_lines:
                    # Find minimum indentation (excluding empty lines)
                    non_empty_lines = [l for l in description_lines if l.strip()]
                    if non_empty_lines:
                        min_indent = min(
                            len(l) - len(l.lstrip()) for l in non_empty_lines
                        )
                        # Remove base indentation and add exactly 4 spaces
                        for desc_line in description_lines:
                            if desc_line.strip():
                                # Remove min_indent, then add 4 spaces
                                new_str_content += (
                                    "\n" + "    " + desc_line[min_indent:]
                                )
                            else:
                                new_str_content += "\n"
                    description_lines = []

                # Split "name (type) – description"
                opened_description_list = False
                left, desc = line.split(" – ", 1)
                param_info = _get_param_raw_info_from_left_string(
                    left, self._current_signature_dict
                )
                formatted_param = self._get_formatted_param(*param_info)
                new_str_content += f"\n\n{formatted_param}\n"
                if desc:
                    new_str_content += f": <!---->\n    {desc}"
                    opened_description_list = True

        # Process any remaining description lines
        if description_lines:
            non_empty_lines = [l for l in description_lines if l.strip()]
            if non_empty_lines:
                min_indent = min(len(l) - len(l.lstrip()) for l in non_empty_lines)
                for desc_line in description_lines:
                    if desc_line.strip():
                        new_str_content += "\n" + "    " + desc_line[min_indent:]
                    else:
                        new_str_content += "\n"

        self.content += new_str_content
        if not (self._prevent_content_stash_deletion):
            self._content_stash = []

    def _restructure_returns_field_body(self):
        pure_str_content = self._content_stash.copy()
        new_str_content = "\n"
        if self._current_return_type:
            new_str_content = f"\n`{self._current_return_type}`"
        new_str_content += "\n: <!---->"

        # Filter out empty lines and find minimum indentation
        non_empty_lines = [c for c in pure_str_content if c.strip()]
        if non_empty_lines:
            min_indent = min(len(c) - len(c.lstrip()) for c in non_empty_lines)
            # Process each line: remove min indent, add 4 spaces
            for c in pure_str_content:
                if c.strip():
                    # Remove the backtick wrapping if present (for backward compatibility)
                    line = (
                        c.lstrip("*   `").rstrip("`")
                        if c.startswith("*   `") and c.endswith("`")
                        else c
                    )
                    # Remove base indentation and add exactly 4 spaces
                    new_str_content += "\n    " + line[min_indent:]
                # Skip empty lines entirely - don't add blank lines within the description

        self.content += new_str_content + "\n\n"
        if not (self._prevent_content_stash_deletion):
            self._content_stash = []

    def _restructure_field_body(self):
        if self._in_parameters_field:
            # If we are in a parameters field, we need to restructure the content
            self._restructure_parameters_field_body()
        elif self._in_returns_field:
            self._restructure_returns_field_body()

    def visit_field_body(self, node):
        self._move_content_to_lines()
        self._prevent_move_content = True

    def depart_field_body(self, node):
        self._restructure_field_body()
        self._prevent_move_content = False
        self.content += "\n"
        self._move_content_to_lines()

    def visit_literal_strong(self, node):
        # Strong literal (e.g., for emphasized code)
        self.content += "**"

    def depart_literal_strong(self, node):
        self.content += "**"

    def visit_title_reference(self, node):
        self.content += "*"  # TODO insert link

    def depart_title_reference(self, node):
        self.content += "*"

    def visit_tabular_col_spec(self, node):
        pass

    def depart_tabular_col_spec(self, node):
        pass

    def visit_autosummary_table(self, node):
        self._table_style = "autosummary"

    def depart_autosummary_table(self, node):
        self._table_style = ""

    def visit_table(self, node):
        pass

    def depart_table(self, node):
        pass

    def visit_tgroup(self, node):
        pass

    def depart_tgroup(self, node):
        pass

    def visit_colspec(self, node):
        pass

    def depart_colspec(self, node):
        pass

    def visit_tbody(self, node):
        pass

    def depart_tbody(self, node):
        pass

    def visit_row(self, node):
        self._entry_count = 0

    def depart_row(self, node):
        pass

    def visit_transition(self, node):
        pass

    def depart_transition(self, node):
        pass

    def visit_entry(self, node):
        self._entry_count += 1
        table_style = getattr(self, "_table_style", "")
        if table_style == "autosummary":
            if self._entry_count == 2:
                while not self.lines[-1]:
                    self.lines.pop()
                self.content += ": <!---->\n    "

    def depart_entry(self, node):
        pass

    def visit_block_quote(self, node):
        pass

    def depart_block_quote(self, node):
        pass
