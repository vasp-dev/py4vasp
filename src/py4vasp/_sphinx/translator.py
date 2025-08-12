# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from typing import Optional

from docutils.nodes import NodeVisitor, SkipNode

from py4vasp._sphinx.anchors_finder import AnchorsFinder
from py4vasp._sphinx.parameters_info_finder import (
    ParametersInfoFinder,
    _get_param_raw_info_from_left_string,
)
from py4vasp._sphinx.return_type_finder import ReturnTypeFinder


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
        self._current_return_type = None
        self._current_signature_dict = {}
        self._prevent_move_content = False
        self._prevent_content_stash_deletion = False
        self._content_stash = []

    def __str__(self):
        return "\n".join(self.lines) + "\n"

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
        print(f"DEBUG: Unknown node type: {node.__class__.__name__}")
        print(f"DEBUG: Node attributes: {node.attributes}")
        print(
            f"DEBUG: Node children: {[child.__class__.__name__ for child in node.children]}"
        )
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

    def visit_title(self, node):
        """Handle title nodes by generating Hugo front matter and Markdown headers.

        The first title becomes Hugo front matter (TOML format) because Hugo expects
        the page title in front matter rather than as a Markdown header. Subsequent
        titles become Markdown headers with the appropriate nesting level based on
        the current section depth.
        """
        self._create_hugo_front_matter(node)
        self.content = f"{self.section_level * '#'} "

    def depart_title(self, node):
        self._move_content_to_lines()

    def _create_hugo_front_matter(self, node):
        """Create Hugo front matter for the document title.
        This method generates the TOML front matter required by Hugo, which includes
        the document title. It is called only once, when the first title node is visited.
        """
        if self.frontmatter_created:
            return
        self.content = f"""\
+++
title = "{node.astext()}"
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

    def depart_section(self, node):
        """Decrement section nesting level when leaving a section.

        This depart method is necessary because we need to restore the previous
        nesting level after processing all content within a section. Without this,
        subsequent sections at the same level would get incorrect header depths.
        """
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
        self.content = "\n" + (self.section_level + 1) * "#" + " ***"

    def depart_rubric(self, node):
        self.content += "***\n\n"
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

    def visit_literal(self, node):
        """Add opening backtick for inline code.

        Single backticks are used for inline code spans in Markdown. This handles
        simple cases where the literal text doesn't contain backticks itself.
        """
        self.content += "`"

    def depart_literal(self, node):
        """Add closing backtick to complete inline code markup.

        Matching backticks are required to properly delimit the code span.
        """
        self.content += "`"

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
        self.content = f"{self.list_stack[-1]:4}"

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
        self._reference_uri = node.get("refuri") or (
            f"#{node.get('refid')}" if node.get("refid") else ""
        )
        self.content += "["

    def depart_reference(self, node):
        """Complete the Markdown link with the stored URI and clean up state.

        The link text has been added by child nodes between visit and depart,
        so we can now close the link with the URI that was stored in visit.
        We clean up the temporary state to avoid interference with other links.
        """
        uri = getattr(self, "_reference_uri", "")
        if (uri) and uri.startswith("#"):
            uri = uri.replace(" ", "-").rstrip("-")
        self.content += f"]({uri})"
        self._reference_uri = None

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
            # Insert an anchor for internal references
            self.content += f'<a name="{refid}"></a>'
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
        """
        # Use reftarget for anchor
        self._reference_uri = f"#{node.get('reftarget', '')}"
        self.content += "["

    def depart_pending_xref(self, node):
        """Complete pending cross-reference links with the stored target.

        Similar to regular references, we close the link with the URI that was
        stored in visit and clean up the temporary state.
        """
        uri = getattr(self, "_reference_uri", "")
        self.content += f"]({uri})"
        self._reference_uri = None

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
        # Start a div to preserve grouping in Markdown
        self.content += '\n<div class="compound">\n\n'

    def depart_compound(self, node):
        # End the div
        self.content += "\n</div>\n"
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
        pass

    def depart_desc(self, node):
        if self.anchor_id_stack:
            self.anchor_id_stack.pop()
        self.section_level -= 1
        self._move_content_to_lines()
        pass

    def _get_parameter_list_and_types(self, node, skip_content: bool = False):
        """Extract parameter names, types, and default values from a desc_signature node."""
        parameters_info_finder = ParametersInfoFinder(self.document)
        parameters_dict = parameters_info_finder.find_parameters_info(
            node, skip_content=skip_content
        )
        parameters = [
            (name, info.get("type"), info.get("default"))
            for name, info in parameters_dict.items()
        ]
        if parameters:
            self._current_signature_dict["sig_parameters"] = parameters.copy()
        return parameters

    def _get_formatted_param(self, name, annotation, default):
        """Format a single parameter with its name, type, and default value."""
        param = f"*{name}*"
        if default or annotation:
            param += ": " + ("[optional] " if default else "")
        if annotation:
            param += f"`{annotation}`"
        if default:
            param += f" [default: {default}]"
        return param

    def _get_parameter_list_str(self, parameters):
        """Get a string representation of the parameter list with types."""
        if not parameters:
            return "()"
        param_strs = []
        for name, annotation, default in parameters:
            param = self._get_formatted_param(name, annotation, default)
            param_strs.append(param)
        if len(param_strs) == 1:
            return f"({param_strs[0]})"
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

    def visit_desc_signature(self, node):
        anchor_id = self._get_anchor_id()
        anchor_str = f"\n\n<a id='{anchor_id}'></a>" if anchor_id else ""
        ref_str = f" [¶](#{anchor_id})" if anchor_id else ""
        objtype = self._get_latest_objtype()
        objtype_str = f"*{objtype}* " if (objtype != "method") else ""
        if objtype:
            self.content += (
                f"\n\n<div class='{f'{objtype} ' if objtype else ''}signature'>"
            )

        name = self._get_latest_name()
        name_str = f"**{name}**"
        self.content += f"{anchor_str}\n\n{self.section_level * '#'} {objtype_str}{name_str}{ref_str}"

        self._current_signature_dict = {}
        self._current_return_type = None
        if objtype in ["function", "method", "class", "exception"]:
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

        self.content += f"\n\n</div>\n\n"

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
        self.content += "\n"
        pass

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
        return f"\n{(self.section_level + 1) * '#'} **{field_name.capitalize()}:**\n\n"

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
        for line in pure_str_content:
            if not (" – " in line):
                if not (opened_description_list):
                    new_str_content += "\n: <!---->"
                    opened_description_list = True
                new_str_content += "\n" + f"    {line}"
            else:
                # Split "name (type) – description"
                opened_description_list = False
                left, desc = line.split(" – ", 1)
                formatted_param = self._get_formatted_param(
                    *(
                        _get_param_raw_info_from_left_string(
                            left, self._current_signature_dict
                        )
                    )
                )
                new_str_content += f"\n\n{formatted_param}\n"
                if desc:
                    new_str_content += f": <!---->\n    {desc}"
                    opened_description_list = True
        self.content = new_str_content
        if not (self._prevent_content_stash_deletion):
            self._content_stash = []

    def _restructure_returns_field_body(self):
        pure_str_content = self._content_stash.copy()
        new_str_content = "\n"
        if self._current_return_type:
            new_str_content = f"\n`{self._current_return_type}`"
        new_str_content += "\n: <!---->"
        new_str_content += "\n    " + "\n    ".join(
            [
                (
                    c.lstrip("*   `").rstrip("`")
                    if c.startswith("*   `") and c.endswith("`")
                    else c
                )
                for c in pure_str_content
            ]
        )
        self.content = new_str_content + "\n\n"
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
        pass

    def visit_literal_strong(self, node):
        # Strong literal (e.g., for emphasized code)
        self.content += "**"

    def depart_literal_strong(self, node):
        self.content += "**"

    def visit_title_reference(self, node):
        self.content += "*"  # TODO insert link

    def depart_title_reference(self, node):
        self.content += "*"
