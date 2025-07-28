# Poisson Regression Test

The Poisson Regression test is a conditional independence test designed for count data. It is frequently employed in constraint-based feature selection to assess whether a variable *X* is associated with a count-based target variable *Y*, given a set of conditioning variables *Z*. Software implementations, such as `testIndPois` in the R package **`MXM`**, leverage this test to identify relevant predictors for count outcomes (Lagani et al., 2017).

## Mathematical Formulation

The test evaluates the null hypothesis that *X* is conditionally independent of *Y* given *Z*. This is done by comparing two nested Poisson regression models using a Likelihood Ratio Test (LRT).

In Poisson regression, the expected count E[*Y*] is modeled via a log link function, assuming that *Y* follows a Poisson distribution (Cameron & Trivedi, 2013).
$$
\ln(E[Y]) = \beta_0 + \beta_1 X_1 + \dots
$$
The two models for the comparison are:

1.  **Restricted Model (Null Hypothesis is true):** This model regresses the count target variable *Y* only on the conditioning set *Z*.
    $$
    H_0: \ln(E[Y]) = \beta_0 + \beta_Z Z
    $$

2.  **Unrestricted Model (Alternative Hypothesis is true):** This model includes both the variable of interest *X* and the conditioning set *Z*.
    $$
    H_A: \ln(E[Y]) = \beta_0 + \beta_X X + \beta_Z Z
    $$

The test statistic *T* is derived from the log-likelihood values of the fitted models:
$$
T = 2 \cdot (\text{log-likelihood}_{\text{unrestricted}} - \text{log-likelihood}_{\text{restricted}})
$$

Based on Wilks's theorem, this statistic *T* is asymptotically distributed as a Chi-Squared (χ²) random variable under the null hypothesis (Wilks, 1938). The degrees of freedom for the χ² distribution equal the difference in the number of parameters between the unrestricted and restricted models (which is 1 when testing a single variable *X*).

## Assumptions

The reliability of the Poisson regression test depends on several key assumptions (Cameron & Trivedi, 2013):

*   **Count Data**: The target variable consists of non-negative integers representing counts.
*   **Independence**: The observations are independent of each other.
*   **Linearity of Log-Rate**: The logarithm of the expected count (the rate) is a linear function of the predictors.
*   **Equidispersion**: A critical assumption of the Poisson model is that the mean and variance of the target variable are equal (E[*Y*] = Var[*Y*]). If the variance is significantly larger than the mean (overdispersion), the standard errors can be underestimated, leading to inflated significance. In such cases, alternatives like the Quasi-Poisson or Negative Binomial regression tests (also available in `MXM`) are more appropriate.

## Code Example

```python
import numpy as np
from citk.tests import Poisson

# Generate data
n = 500
X = np.random.randn(n)
Z = 0.5 * X + np.random.randn(n)
# Y is conditionally dependent on Z, but not on X given Z
Y = np.random.poisson(np.exp(1 + 0.5 * Z))
data = np.vstack([X, Y, Z]).T

# Initialize the test
poisson_test = Poisson(data)

# Test for conditional independence of X and Y given Z
p_value = poisson_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value:.4f}")
```

## API Reference

:class:`citk.tests.statistical_model_tests.Poisson`

## References

*   Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press.
*   Lagani, V., Athineou, G., Farcomeni, A., Tsagris, M., & Tsamardinos, I. (2017). Feature Selection with the R Package MXM: Discovering Statistically Equivalent Feature Subsets. *Journal of Statistical Software, 80*(7), 1-25.
*   Wilks, S. S. (1938). The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses. *The Annals of Mathematical Statistics, 9*(1), 60–62. 