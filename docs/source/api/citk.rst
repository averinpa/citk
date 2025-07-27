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

Statistical Model-Based Tests
-----------------------------

These tests use statistical regression models to test for conditional independence.

.. autoclass:: citk.tests.Regression
   :show-inheritance:

.. autoclass:: citk.tests.Logit
   :show-inheritance:

.. autoclass:: citk.tests.Poisson
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
