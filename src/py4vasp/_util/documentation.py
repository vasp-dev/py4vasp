def _add_documentation(doc_string):
    def add_documentation_to_function(func):
        func.__doc__ = doc_string
        return func

    return add_documentation_to_function
