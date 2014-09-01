Freeflight Analysis
===================

This module contains code for working with freeflight data. It is designed
to let you create both interactive and programmatic analysis scripts.

For interactive commandline tools you should use
:py:func:`analysislib.args.get_parser` and :py:meth:`analysislib.combine.CombineH5WithCSV.add_from_args`.

For programmatic scripts you should use

:py:func:`analysislib.util.get_combiner` and 
:py:meth:`analysislib.combine.CombineH5WithCSV.get_results` or
:py:meth:`analysislib.combine.CombineH5WithCSV.get_one_result`. Or simply
:py:func:`analysislib.util.get_one_trajectory`.


Combining Data
==============

.. autoclass:: analysislib.combine.CombineH5WithCSV
   :member-order: groupwise
   :inherited-members:

Programmatic Scripts
====================

.. automodule:: analysislib.util
   :members:

Command Line Tools
==================

Followpath Monitor

draws the path the fly should follow (e.g. the xx.svg file in your stimulus), the position of the fly and the position it shuld go next
it is located in the folder "nodes"
   rosrun strawlab_freeflight_experiments followpath_monitor.py
.. automodule:: analysislib.args
   :members:


