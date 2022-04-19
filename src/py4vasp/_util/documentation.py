# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)


def add(doc_string):
    def add_documentation_to_function(func):
        func.__doc__ = doc_string
        return func

    return add_documentation_to_function
