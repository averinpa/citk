# Conformalized Residual Independence Test (CRIT)

The Conformalized Residual Independence Test (CRIT) is a sophisticated, non-parametric method for conditional independence testing. It enhances the standard residual-based approach by integrating quantile regression with the rigorous statistical guarantees of **conformal prediction**. This combination allows CRIT to produce robust, distribution-free residuals, making the subsequent independence test more reliable, especially in the presence of complex, non-linear relationships and heteroscedastic noise.

## Mathematical Formulation

CRIT is built on the principle that if *X* and *Y* are independent given *Z*, then the parts of *X* and *Y* that cannot be explained by *Z* should also be independent. The novelty of CRIT lies in how it rigorously defines and calculates these "unexplained parts" or residuals. The process can be broken down into four main stages:

1.  **Quantile Regression for Prediction Intervals**: Instead of predicting only the conditional mean (e.g., E[*Y*|*Z*]), CRIT uses quantile regression to model the conditional distribution more fully. For a chosen significance level α (e.g., α=0.1), two models are trained to predict the conditional lower and upper quantiles of *Y* given *Z*.
    *   $\hat{q}_{\alpha/2}(Z) \approx$ the $(\alpha/2)$-quantile of $Y$ given $Z$
    *   $\hat{q}_{1-\alpha/2}(Z) \approx$ the $(1-\alpha/2)$-quantile of $Y$ given $Z$
    These two quantile functions define an initial prediction interval $[\hat{q}_{\alpha/2}(Z), \hat{q}_{1-\alpha/2}(Z)]$ that is intended to contain the true value of *Y* with a probability of 1-α. The same procedure is applied to predict an interval for *X* given *Z*.

2.  **Conformalization for Valid Coverage**: A key issue with standard prediction intervals is that their actual coverage rate may not match the desired rate (1-α) if the quantile regression models are misspecified. Conformal prediction is a technique that corrects these intervals to provide a mathematically guaranteed coverage rate (Vovk et al., 2005). Using a separate calibration dataset, a non-conformity score is calculated for each point, typically measuring how far the true value lies outside its predicted interval. These scores are then used to adjust the width of the prediction intervals for new data, ensuring they achieve the desired coverage in the long run.

3.  **Residual Calculation**: CRIT introduces a novel way of calculating residuals. Once a calibrated prediction interval is obtained for an observation *y*, its residual is not its distance from the mean, but its relative position within this interval. The residual, *R<sub>Y</sub>*, is defined as:
    $$
    R_Y = \frac{y - \hat{c}(Z)}{\hat{s}(Z)}
    $$
    where $\hat{c}(Z)$ is the center of the calibrated interval and $\hat{s}(Z)$ is its width. This calculation effectively transforms the original variable into a new one whose values are standardized based on the conditional distribution given *Z*. A similar residual, *R<sub>X</sub>*, is calculated for *X*.

4.  **Final Independence Test**: After obtaining the conformalized residuals *R<sub>X</sub>* and *R<sub>Y</sub>*, the final step is to test whether they are unconditionally independent. The null hypothesis of the original test ($X \perp Y | Z$) is now translated into a simpler null hypothesis ($R_X \perp R_Y$). This implementation performs a **distance correlation** test on these residuals (Székely et al., 2007). Distance correlation is a powerful non-parametric measure that is zero if and only if the variables are independent.

## Properties and Assumptions

*   **Distribution-Free**: Thanks to the guarantees of conformal prediction, the resulting test is valid (i.e., controls the Type I error rate) without requiring strong assumptions about the data's distribution.
*   **Robustness**: The method is robust to heteroscedasticity (non-constant variance of errors) and misspecification of the underlying regression models used to predict the quantiles.
*   **Assumptions**: The primary assumption is that the quantile regression models are reasonably well-specified to capture the conditional distributions. The performance depends on the quality of these base models. The data is also assumed to be exchangeable.

## Code Example

```python
import numpy as np
from citk.tests import CRIT

# Generate data with a non-linear relationship: X -> Z -> Y
n = 500
X = np.random.randn(n)
Z = np.sin(X * 2) + np.random.randn(n) * 0.2
Y = Z**3 + np.random.randn(n) * 0.2
data = np.vstack([X, Y, Z]).T

# Initialize the test
crit_test = CRIT(data, alpha=0.1, n_perms=99)

# Test for unconditional independence (should be dependent)
p_unconditional = crit_test(0, 1)
print(f"P-value (unconditional) for X _||_ Y: {p_unconditional:.4f}")

# Test for conditional independence given Z (should be independent)
p_conditional = crit_test(0, 1, [2])
print(f"P-value (conditional) for X _||_ Y | Z: {p_conditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.ml_based_tests.CRIT`.

## References

*   Székely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007). Measuring and testing dependence by correlation of distances. *The Annals of Statistics, 35*(6), 2769-2794.
*   Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic learning in a random world. *Springer Science & Business Media*. 