.. _citk-tests:

citk.tests package
==================

This package contains the implementations of various conditional independence tests.

Regression-Based Tests
----------------------

.. autoclass:: citk.tests.tigramite_based_tests.RegressionCI
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

Machine Learning-Based Tests
----------------------------

.. autoclass:: citk.tests.ml_based_tests.KCI
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.ml_based_tests.RandomForest
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.ml_based_tests.DML
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.ml_based_tests.CRIT
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.ml_based_tests.EDML
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

Simple Correlation-Based Tests
------------------------------

.. autoclass:: citk.tests.simple_tests.G_sq
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.simple_tests.Chi_sq
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.simple_tests.Spearman
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.simple_tests.Fisher_Z
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__
