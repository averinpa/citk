API Reference
=============

This page provides a detailed API reference for the conditional independence tests available in `citk`.

Simple CI Tests
---------------

These are classical statistical methods for conditional independence testing.

.. autoclass:: citk.tests.FisherZ
   :show-inheritance:

.. autoclass:: citk.tests.Spearman
   :show-inheritance:

.. autoclass:: citk.tests.ChiSq
   :show-inheritance:

.. autoclass:: citk.tests.GSq
   :show-inheritance:

Regression-Based Tests
----------------------

These tests use regression-based formulations for conditional independence.

.. autoclass:: citk.tests.RegressionCI
   :show-inheritance:

Machine Learning-Based Tests
----------------------------

These tests leverage machine learning models to detect complex, non-linear conditional dependencies.

.. autoclass:: citk.tests.KCI
   :show-inheritance:

.. autoclass:: citk.tests.RandomForest
   :show-inheritance:

.. autoclass:: citk.tests.DML
   :show-inheritance:

.. autoclass:: citk.tests.CRIT
   :show-inheritance:

.. autoclass:: citk.tests.EDML
   :show-inheritance:
