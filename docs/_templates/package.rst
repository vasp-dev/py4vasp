{{ name | escape | underline }}

.. automodule:: {{ fullname }}

Attributes
----------
.. autosummary::
  :toctree:

  {% for member in members %}
  {{ member }}
  {% endfor %}
