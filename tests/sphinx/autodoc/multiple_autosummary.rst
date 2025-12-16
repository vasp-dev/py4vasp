Multiple autosummary
====================

This tests whether multiple autosummary directives are correctly processed.

.. autosummary::
    :nosignatures:

    ~example_class.ExampleClass

Text in between.

.. autosummary::
    :nosignatures:

    ~example_module.ExampleModule

Text in between.

.. autosummary::
    :nosignatures:

    ~example_module.ExampleDataClass