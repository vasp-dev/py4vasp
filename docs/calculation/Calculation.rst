Calculation
===========

.. autoclass:: py4vasp.Calculation
   :members: from_path, from_file, path
   
Available quantities
--------------------

.. jinja::
    .. autosummary::
        :nosignatures:
    {% for quantity in calculation.QUANTITIES %}
        {% if not quantity.startswith("_") -%}
            ~py4vasp.Calculation.{{ quantity }}
        {%- endif -%}
    {% endfor %}
