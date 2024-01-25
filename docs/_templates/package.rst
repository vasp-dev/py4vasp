{{ name | escape | underline }}

.. automodule:: {{ fullname }}

Attributes
----------
.. autosummary::
  :toctree:
  :template: member.rst

  {% for member in members %}
  {{ member }}
  {% endfor %}
