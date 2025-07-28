# E-Value Double Machine Learning (EDML) CI Test

The E-Value Double Machine Learning (EDML) test is a modern framework for conditional independence testing that replaces the traditional p-value with an **e-value**. An e-value is a measure of statistical evidence against the null hypothesis that offers several profound advantages, particularly for complex, iterative tasks like causal discovery.

## Mathematical Formulation

The EDML test integrates the robust residualization procedure of Double Machine Learning with the modern inference framework of e-values. The process unfolds in two primary stages:

1.  **Nuisance Prediction and Residualization**: This stage is identical to the standard DML procedure (Chernozhukov et al., 2018). The goal is to isolate the parts of *X* and *Y* that are not explained by the conditioning set *Z*.
    *   Two flexible machine learning models are trained using cross-fitting to predict *X* from *Z* and *Y* from *Z*.
        *   $X = f(Z) + U$
        *   $Y = g(Z) + V$
    *   The resulting residuals, *U* and *V*, represent the variation in *X* and *Y* that is orthogonal to *Z*. Under the null hypothesis ($X \perp Y | Z$), these residuals should be independent.

2.  **E-Value Calculation from Residuals**: Instead of performing a standard hypothesis test on the residuals to obtain a p-value, EDML constructs an e-value. This is achieved through a sequential betting mechanism, formally known as a **test martingale**.
    *   A simple predictive model (the "betting strategy," e.g., Ridge regression) is trained on one portion of the residuals. This model is then used to sequentially "bet" on the relationship in the remaining portion. The final e-value, *E*, is the product of the outcomes of these sequential bets.
    *   An e-value greater than 1 provides evidence against the null hypothesis. By Markov's inequality, an e-value *E* can be converted to a p-value that controls the Type I error rate, typically via the simple formula **p = 1/E**. An e-value of 20, for instance, corresponds to a p-value of 0.05.

## Advantages for Causal Discovery

While traditional p-values are standard, their properties can be problematic for causal discovery algorithms (like the PC algorithm) which are highly iterative and adaptive. E-values provide two key advantages in this setting:

1.  **Validity in Adaptive Search Procedures:** Causal discovery algorithms are inherently adaptive; the result of one CI test (e.g., $X \perp Y$) determines which test to perform next (e.g., $X \perp Y | Z$). The formal guarantees of p-values can be challenged by this sequential, data-dependent process. E-values, constructed as test martingales, are naturally suited for this sequential framework, providing more robust statistical guarantees throughout the adaptive search for a causal graph (Grünwald & Shafer, 2021).

2.  **Flexible Integration of Evidence:** There is often no single CI test that is best for all scenarios. A researcher may want to combine the results from a fast, linear CI test with a more computationally intensive, non-linear test for the same potential edge. Combining p-values is non-trivial and requires special methods. E-values, however, can be simply multiplied. This allows an algorithm to easily integrate evidence from different test types (or even different datasets) by multiplying their e-values, yielding a single, aggregated measure of evidence for or against an independence.

## Assumptions

*   Inherits the primary assumption from DML: the machine learning models used for residualization must be flexible enough to consistently estimate the true nuisance functions.
*   The betting mechanism (test martingale) used to generate the e-value must be valid, meaning its expected value is 1 under the null hypothesis.

## Code Example

```python
import numpy as np
from citk.tests import EDML

# Generate data with a linear relationship: X -> Z -> Y
n = 1000
X = np.random.randn(n)
Z = 2 * X + np.random.randn(n) * 0.5
Y = 3 * Z + np.random.randn(n) * 0.5
data = np.vstack([X, Y, Z]).T

# Initialize the test.
edml_test = EDML(data)

# Test for unconditional independence (should be dependent, p-value should be small)
p_unconditional = edml_test(0, 1)
print(f"P-value (unconditional) for X _||_ Y: {p_unconditional:.4f}")

# Test for conditional independence given Z (should be independent, p-value should be large)
p_conditional = edml_test(0, 1, [2])
print(f"P-value (conditional) for X _||_ Y | Z: {p_conditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.ml_based_tests.EDML`.

## References

*   Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal, 21*(1), C1-C68.
*   Grünwald, P., & Shafer, G. (2021). E-Values: A Guide for the Uninitiated. *arXiv preprint arXiv:2111.08272*. 