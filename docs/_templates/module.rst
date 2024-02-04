{{ name | escape | underline }}

.. automodule:: {{ fullname }}

.. autosummary::

   {% for function in functions %}
   {{ function }}
   {% endfor %}

{% for function in functions %}
.. autofunction:: {{ function }}
{% endfor %}
