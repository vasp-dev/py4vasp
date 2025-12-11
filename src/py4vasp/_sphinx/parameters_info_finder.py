# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from docutils.nodes import NodeVisitor, SkipNode

SIG_TYPE_DEFAULT = "`?_UNKNOWN_?`"


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


def _normalize_param_name(name: str) -> str:
    """Remove all asterisks from parameter name for comparison purposes.

    Examples:
        '*args' -> 'args'
        '**kwargs' -> 'kwargs'
        '*args*' -> 'args'
        'normal' -> 'normal'
    """
    return name.strip("*")


def _extract_leading_asterisks(name: str) -> str:
    """Extract leading asterisks from parameter name (unpacking operators).

    Examples:
        '*args' -> '*'
        '**kwargs' -> '**'
        '*args*' -> '*'
        'normal' -> ''
    """
    count = 0
    for char in name:
        if char == "*":
            count += 1
        else:
            break
    return "*" * count


class ParametersInfoFinder(NodeVisitor):
    """A docutils NodeVisitor that finds parameters info in Sphinx document trees."""

    def __init__(self, document):
        super().__init__(document)
        self._parameters = {}
        self._in_parameters_field = False
        self._in_signature = False
        self._next_is_type = False
        self._next_is_default = False
        self._current_parameter_name: str = ""
        self._skip_content = False

    def __str__(self):
        return "\n".join(self.lines) + "\n"

    def find_parameters_info(self, node, skip_content: bool = False) -> dict:
        """Retrieves parameter information from a desc_signature node.

        Parameters
        ----------
        node : docutils.nodes.Node
            The desc_signature node to retrieve parameter information from.
        skip_content : bool, optional
            Whether to skip content processing.
            This should be true only when calling from function or method objtypes, otherwise False to prevent retrieving information from deeper layers.

        Returns
        -------
        dict
            A dictionary containing parameter information: { "name": { "type": my_type, "default": my_default } }
        """
        if node.__class__.__name__ != "desc_signature":
            raise UserWarning(
                "Node passed to get_parameter_list_and_types is not a desc_signature node. Parameters not retrieved."
            )
        else:
            self._skip_content = skip_content
            self._parameters = {}
            node.parent.walkabout(self)
        return self._parameters

    def unknown_visit(self, node):
        raise SkipNode

    def unknown_departure(self, node):
        """Handle departure from nodes that don't have specific depart methods.

        Most Markdown constructs don't require closing syntax, so we provide this
        no-op handler to avoid errors when the visitor tries to call depart methods
        that don't exist.
        """
        pass

    def visit_desc_signature(self, node):
        self._in_signature = True

    def depart_desc_signature(self, node):
        self._in_signature = False

    def visit_desc_parameterlist(self, node):
        pass

    def visit_desc_parameter(self, node):
        self._current_parameter_name = ""
        self._next_is_type = False
        self._next_is_default = False

    def visit_desc_sig_name(self, node):
        if not (self._next_is_type):
            raw_name = node.astext()

            # Check if there's a desc_sig_operator sibling before this node (for * or **)
            leading_asterisks = ""
            parent = node.parent
            if parent and hasattr(parent, "children"):
                for child in parent.children:
                    if child == node:
                        break  # Stop when we reach the current node
                    if child.tagname == "desc_sig_operator":
                        op_text = child.astext()
                        if op_text in ("*", "**"):
                            leading_asterisks = op_text

            normalized_name = _normalize_param_name(raw_name)

            # Store with normalized name but keep asterisks
            self._current_parameter_name = normalized_name
            self._next_is_type = True

            if not (self._parameters.get(normalized_name)):
                self._parameters[normalized_name] = {}

            # Store the asterisks if present
            if leading_asterisks:
                self._parameters[normalized_name]["asterisks"] = leading_asterisks
        elif self._current_parameter_name:
            if self._parameters[self._current_parameter_name].get("type") == None:
                self._parameters[self._current_parameter_name]["type"] = (
                    _extract_type_text(node)
                )
            self._next_is_type = False

    def visit_desc_sig_operator(self, node):
        if "=" in node.astext():
            self._next_is_default = True

    def visit_desc_sig_punctuation(self, node):
        if "=" in node.astext():
            self._next_is_default = True

    def depart_desc_parameter(self, node):
        self._current_parameter_name = ""
        self._next_is_type = False
        self._next_is_default = False

    def visit_inline(self, node):
        if self._next_is_default and self._current_parameter_name:
            if self._parameters[self._current_parameter_name].get("default") == None:
                self._parameters[self._current_parameter_name][
                    "default"
                ] = node.astext()
            self._next_is_default = False

    def visit_desc_content(self, node):
        if self._skip_content:
            raise SkipNode
        pass

    def visit_desc(self, node):
        pass

    def visit_field_list(self, node):
        pass

    def visit_field(self, node):
        pass

    def visit_field_name(self, node):
        if node.astext().lower() == "parameters":
            self._in_parameters_field = True
        raise SkipNode

    def visit_field_body(self, node):
        if self._in_parameters_field:
            pass
        else:
            raise SkipNode

    def depart_field(self, node):
        self._in_parameters_field = False
        pass

    def _add_param_info_from_node(self, node):
        lines = node.astext().splitlines()
        for line in lines:
            if " – " in line:
                left, _ = line.split(" – ", 1)
                name, type_, default = _get_param_raw_info_from_left_string(
                    left, _construction_dict=self._parameters
                )
                # Normalize the name for storage
                normalized_name = _normalize_param_name(name)

                if normalized_name not in self._parameters:
                    self._parameters[normalized_name] = {}

                # Preserve asterisks if present in the returned name
                asterisks = _extract_leading_asterisks(name)
                if asterisks and not self._parameters[normalized_name].get("asterisks"):
                    self._parameters[normalized_name]["asterisks"] = asterisks

                if self._parameters[normalized_name].get("type") == None:
                    self._parameters[normalized_name]["type"] = type_
                if self._parameters[normalized_name].get("default") == None:
                    self._parameters[normalized_name]["default"] = default

    def visit_paragraph(self, node):
        if self._in_parameters_field:
            self._add_param_info_from_node(node)

    def visit_bullet_list(self, node):
        if not (self._in_parameters_field):
            raise SkipNode

    def visit_list_item(self, node):
        if not (self._in_parameters_field):
            raise SkipNode
        else:
            self._add_param_info_from_node(node)


def _get_param_raw_info_from_left_string(
    left: str, _current_signature_dict: dict = {}, _construction_dict: dict = {}
) -> tuple[str, str, str]:
    """Extract parameter name, type, and default value from the left part of a field.

    Returns the name WITH asterisks preserved from signature."""
    sep_around_type = left.split(" (")
    # Strip emphasis asterisks (*name*) but preserve leading asterisks for matching
    raw_name = sep_around_type[0].strip("* ")
    normalized_name = _normalize_param_name(raw_name)

    sig_default = None
    sig_types = None
    sig_asterisks = ""

    if _current_signature_dict.get("sig_parameters"):
        # Check if the name is found in sig_parameters (using normalized names)
        for param_info in _current_signature_dict["sig_parameters"]:
            param, _type, default = param_info
            # Also check if param is a dict with asterisks info
            if isinstance(param, dict):
                param_asterisks = param.get("asterisks", "")
                param_name = param.get("name", "")
            else:
                param_asterisks = _extract_leading_asterisks(param)
                param_name = _normalize_param_name(param)

            if param_name == normalized_name:
                sig_default = default
                sig_types = _type
                sig_asterisks = param_asterisks
                break
    elif _construction_dict:
        if _construction_dict.get(normalized_name):
            sig_default = _construction_dict[normalized_name].get("default", None)
            sig_types = _construction_dict[normalized_name].get("type", None)
            sig_asterisks = _construction_dict[normalized_name].get("asterisks", "")

    # Reconstruct the name with asterisks from signature
    final_name = sig_asterisks + normalized_name if sig_asterisks else normalized_name

    type_annotation = None
    if len(sep_around_type) > 1 or sig_types:
        # If we have a type annotation, format it
        type_annotation = sig_types
        if len(sep_around_type) > 1:
            # Use rsplit to handle markdown links like [text](#anchor) correctly
            # This splits from the right, getting the last ) which is the closing paren of the type
            types_part = sep_around_type[1].rsplit(")", 1)
            if len(types_part) > 1:
                types = types_part[0].strip()
            else:
                # No closing paren found, use the whole string
                types = sep_around_type[1].strip()
            types_and_optional = types.split(",")
            pure_types = [
                t.strip(" `") for t in types_and_optional if not ("optional" in t)
            ]
            type_annotation = " | ".join(pure_types)
            if (sig_default == None) and ("optional" in types):
                sig_default = SIG_TYPE_DEFAULT
    return final_name, type_annotation, sig_default
