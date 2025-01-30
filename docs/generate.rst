generate
========

.. jinja::

    {% for quantity in calculation.QUANTITIES %}
        {% if not quantity.startswith("_") %}
            .. automodule:: py4vasp._calculation.{{ quantity }}
                :members:
        {% endif %}
    {% endfor %}

