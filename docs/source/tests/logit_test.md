# Logistic Regression Test

The Logistic Regression test is a conditional independence test for binary or categorical target variables. It is a fundamental tool in constraint-based feature selection, used to determine if a variable *X* provides statistically significant information about a target *Y*, after accounting for a set of conditioning variables *Z*. This test is implemented in software packages like **`MXM`** (as `testIndLogistic`) to facilitate feature selection from high-dimensional datasets (Lagani et al., 2017).

## Mathematical Formulation

The test assesses the null hypothesis that *X* is conditionally independent of *Y* given *Z*. This is achieved by comparing the goodness-of-fit of two nested logistic regression models using a Likelihood Ratio Test (LRT).

In logistic regression, the model predicts the probability of the outcome *Y* being 1, P(Y=1), via the logit (log-odds) link function:
$$
\text{logit}(P(Y=1)) = \ln\left(\frac{P(Y=1)}{1-P(Y=1)}\right) = \beta_0 + \beta_1 X_1 + \dots
$$
The two models compared are:

1.  **Restricted Model (Null Hypothesis is true):** This model regresses the binary target variable *Y* only on the conditioning set *Z*.
    $$
    H_0: \text{logit}(P(Y=1)) = \beta_0 + \beta_Z Z
    $$

2.  **Unrestricted Model (Alternative Hypothesis is true):** This model includes both the variable of interest *X* and the conditioning set *Z*.
    $$
    H_A: \text{logit}(P(Y=1)) = \beta_0 + \beta_X X + \beta_Z Z
    $$

The test statistic *T* is calculated from the log-likelihood values of the fitted models:
$$
T = 2 \cdot (\text{log-likelihood}_{\text{unrestricted}} - \text{log-likelihood}_{\text{restricted}})
$$

According to Wilks's theorem, under the null hypothesis of conditional independence (i.e., that the coefficient *β<sub>X</sub>* is zero), this statistic *T* asymptotically follows a Chi-Squared (χ²) distribution (Wilks, 1938). The degrees of freedom are equal to the difference in the number of parameters between the two models (typically 1 when testing a single variable *X*).

## Assumptions

The validity of the test relies on several assumptions of logistic regression (Hosmer et al., 2013):

*   **Binary or Categorical Outcome**: The target variable is binary (e.g., 0/1, pass/fail) or categorical. The `MXM` implementation also extends this to multinomial and ordinal outcomes.
*   **Independence of Observations**: The observations in the dataset are assumed to be independent of each other.
*   **Linearity of Log-Odds**: The relationship between the predictors and the log-odds of the outcome is assumed to be linear.
*   **Absence of Strong Multicollinearity**: The predictor variables should not be highly correlated with each other, as this can inflate the variance of the coefficient estimates.

## Code Example

```python
import numpy as np
from citk.tests import Logit

# Generate data
n = 500
X = np.random.randn(n)
Z = 2 * X + np.random.randn(n)
Y = (3 * Z + np.random.randn(n)) > 0
data = np.vstack([X, Y, Z]).T

# Initialize the test
logit_test = Logit(data)

# Test for conditional independence of X and Y given Z
p_value = logit_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value:.4f}")
```

## API Reference

:class:`citk.tests.statistical_model_tests.Logit`

## References

*   Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.
*   Lagani, V., Athineou, G., Farcomeni, A., Tsagris, M., & Tsamardinos, I. (2017). Feature Selection with the R Package MXM: Discovering Statistically Equivalent Feature Subsets. *Journal of Statistical Software, 80*(7), 1-25.
*   Wilks, S. S. (1938). The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses. *The Annals of Mathematical Statistics, 9*(1), 60–62. 