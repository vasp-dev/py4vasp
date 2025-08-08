Footnotes
=========

This is a test for footnotes in Sphinx documentation. [#first_footnote]_
This is a second footnote reference. [#second_footnote]_

This is a second paragraph with another footnote. [#third_footnote]_

.. [#first_footnote] This is the first footnote.
   It can contain multiple lines of text and even
   some formatting like *bold* or **italic**.

.. [#second_footnote] This is the second footnote.

  The second footnote has multiple paragraphs and a code block:

  .. code-block:: python

     print("This is a code block in a footnote.")

.. [#third_footnote] This is the third footnote with a definition list.

   term
       Definition of the term in the third footnote.
   
   next term
       Description of the next term in the third footnote.
