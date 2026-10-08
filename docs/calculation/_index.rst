calculation
===========

.. autodata:: py4vasp.calculation
   :annotation:

Available quantities
--------------------

.. jinja::
    .. autosummary::
        :nosignatures:
    {% for autosummary in calculation.AUTOSUMMARIES %}
        {{ autosummary[1] }}
    {% endfor %}

Constants
---------

.. autosummary::
    :nosignatures:

    ~py4vasp._calculation.mass.Mass