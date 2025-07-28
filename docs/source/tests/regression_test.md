# Linear Regression Test

The Linear Regression test is a conditional independence test for continuous data. In the context of feature selection, it determines whether a variable of interest, *X*, is independent of a target variable, *Y*, given a set of conditioning variables, *Z*. This test is a core component of many modern causal discovery and feature selection algorithms, where it is used to prune statistically irrelevant variables. For instance, the R package **`MXM`** (Mens eX Machina) implements this test as `testIndReg` for its constraint-based algorithms (Lagani et al., 2017).

## Mathematical Formulation

The test evaluates the null hypothesis that *X* is conditionally independent of *Y* given *Z*. This is done by comparing two nested Ordinary Least Squares (OLS) models:

1.  **Restricted Model (Null Hypothesis is true):** This model regresses the target variable *Y* only on the conditioning set *Z*.
    $$
    H_0: Y = \beta_0 + \beta_Z Z + \epsilon
    $$

2.  **Unrestricted Model (Alternative Hypothesis is true):** This model regresses *Y* on both the variable of interest *X* and the conditioning set *Z*.
    $$
    H_A: Y = \beta_0 + \beta_X X + \beta_Z Z + \epsilon
    $$

The core idea is to assess whether adding *X* to the model significantly improves its fit to the data. This is most commonly done using an **F-test**, which compares the residual sum of squares (RSS) of the two models (Kutner et al., 2005).

The F-statistic is calculated as:
$$
F = \frac{(\text{RSS}_{\text{restricted}} - \text{RSS}_{\text{unrestricted}}) / q}{\text{RSS}_{\text{unrestricted}} / (n - k - 1)}
$$

Where:
*   **RSS<sub>restricted</sub>** is the residual sum of squares of the model without *X*.
*   **RSS<sub>unrestricted</sub>** is the residual sum of squares of the model with *X*.
*   **q** is the number of parameters being tested (in this simple case, *q*=1).
*   **n** is the number of observations.
*   **k** is the number of predictors in the unrestricted model.

Under the null hypothesis, this F-statistic follows an F-distribution with *q* and *(n - k - 1)* degrees of freedom. A high F-value (and a correspondingly low p-value) provides evidence to reject the null hypothesis, suggesting that *X* has a statistically significant linear relationship with *Y*, even after accounting for *Z*.

Alternatively, a **Likelihood Ratio Test (LRT)** can be used, which is the general approach for the GLM-based tests in `MXM`. For OLS with the assumption of normally distributed errors, the LRT is equivalent to the F-test. The test statistic is calculated from the log-likelihood of the models:
$$
T = 2 \cdot (\text{log-likelihood}_{\text{unrestricted}} - \text{log-likelihood}_{\text{restricted}})
$$
According to Wilks's theorem, this statistic asymptotically follows a Chi-Squared (χ²) distribution with *q* degrees of freedom under the null hypothesis (Wilks, 1938).

## Assumptions

For the F-test and associated p-values to be reliable, several OLS assumptions should be met (Kutner et al., 2005):

*   **Linearity**: The relationship between the predictors and the target variable is linear.
*   **No Perfect Multicollinearity**: The independent variables should not be perfectly correlated.
*   **Homoscedasticity**: The variance of the errors is constant for all levels of the predictors.
*   **Independence of Errors**: The errors are uncorrelated with each other.
*   **Normality of Errors**: For statistical inference in smaller samples, the errors are assumed to follow a normal distribution. While OLS estimates remain unbiased without normality, hypothesis testing relies on this assumption.

## Code Example

```python
import numpy as np
from citk.tests import Regression

# Generate data
n = 500
X = np.random.randn(n)
Z = 2 * X + np.random.randn(n)
Y = 3 * Z + np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Initialize the test
regression_test = Regression(data)

# Test for conditional independence of X and Y given Z
p_value = regression_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value:.4f}")
```

## API Reference

:class:`citk.tests.statistical_model_tests.Regression`

## References

*   Kutner, M. H., Nachtsheim, C. J., Neter, J., & Li, W. (2005). *Applied Linear Statistical Models* (5th ed.). McGraw-Hill Irwin.
*   Lagani, V., Athineou, G., Farcomeni, A., Tsagris, M., & Tsamardinos, I. (2017). Feature Selection with the R Package MXM: Discovering Statistically Equivalent Feature Subsets. *Journal of Statistical Software, 80*(7), 1-25.
*   Wilks, S. S. (1938). The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses. *The Annals of Mathematical Statistics, 9*(1), 60–62. 