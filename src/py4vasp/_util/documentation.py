# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import inspect


def add(doc_string):
    def add_documentation_to_function(func):
        func.__doc__ = doc_string
        return func

    return add_documentation_to_function


def format(**kwargs):
    def format_documentation_of_function(func):
        clean_kwargs = {key: str(value).rstrip() for key, value in kwargs.items()}
        func.__doc__ = inspect.getdoc(func).format(**clean_kwargs)
        return func

    return format_documentation_of_function
