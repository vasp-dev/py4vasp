# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from docutils.nodes import NodeVisitor, SkipNode


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
            self._current_parameter_name = node.astext()
            self._next_is_type = True
            if not (self._parameters.get(self._current_parameter_name)):
                self._parameters[self._current_parameter_name] = {}
        elif self._current_parameter_name:
            if self._parameters[self._current_parameter_name].get("type") == None:
                self._parameters[self._current_parameter_name]["type"] = node.astext()
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
                if name not in self._parameters:
                    self._parameters[name] = {}

                if self._parameters[name].get("type") == None:
                    self._parameters[name]["type"] = type_
                if self._parameters[name].get("default") == None:
                    self._parameters[name]["default"] = default

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
    """Extract parameter name, type, and default value from the left part of a field."""
    sep_around_type = left.split(" (")
    left = sep_around_type[0].strip("* ")
    sig_default = None
    sig_types = None
    if _current_signature_dict.get("sig_parameters"):
        # Check if the name is found in sig_parameters
        for param, _type, default in _current_signature_dict["sig_parameters"]:
            if param == left:
                sig_default = default
                sig_types = _type
                break
    elif _construction_dict:
        if _construction_dict.get(left):
            sig_default = _construction_dict[left].get("default", None)
            sig_types = _construction_dict[left].get("type", None)

    type_annotation = None
    if len(sep_around_type) > 1 or sig_types:
        # If we have a type annotation, format it
        type_annotation = sig_types
        if len(sep_around_type) > 1:
            types = sep_around_type[1].split(")")[0].strip()
            types_and_optional = types.split(",")
            pure_types = [
                t.strip(" `") for t in types_and_optional if not ("optional" in t)
            ]
            type_annotation = " | ".join(pure_types)
            if (sig_default == None) and ("optional" in types):
                sig_default = "`?_UNKNOWN_?`"
    return left, type_annotation, sig_default
