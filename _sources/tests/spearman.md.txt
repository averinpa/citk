# Spearman's Rho Test

The Spearman's Rho test is a non-parametric conditional independence test for continuous data, first introduced by Spearman (1904). It is a robust alternative to the Fisher's Z test, particularly when the assumption of linearity is not met.

## Mathematical Formulation

The test operates by first converting the data to ranks. It then calculates the partial Pearson correlation on the ranked data, which is equivalent to Spearman's partial correlation, $r_s = \rho(R(X), R(Y) | R(Z))$, where $R(V)$ is the rank of variable $V$.

The test statistic is then derived using the Fisher's Z-transformation on this rank-based correlation (Kendall & Stuart, 1973):

```{math}
Z(r_s) = \frac{1}{2} \ln\left(\frac{1+r_s}{1-r_s}\right)
```

The final test statistic is:

```{math}
T = \sqrt{n - |Z| - 3} \cdot |Z(r_s)|
```

where $n$ is the sample size and $|Z|$ is the number of conditioning variables. This statistic follows a standard normal distribution, $N(0, 1)$.

## Assumptions

- The relationship between variables is monotonic (either consistently increasing or decreasing).
- It does not assume a linear relationship or a multivariate normal distribution.

## Code Example

```python
import numpy as np
from citk.tests import Spearman

# Generate data with a non-linear, monotonic relationship
# X -> Z -> Y
n = 500
X = np.random.rand(n) * 5
Z = np.exp(X / 2) + np.random.randn(n) * 0.1
Y = np.log(Z**2) + np.random.randn(n) * 0.1
data = np.vstack([X, Y, Z]).T

# Initialize the test
spearman_test = Spearman(data)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = spearman_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = spearman_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.simple_tests.Spearman`.

## References

Spearman, C. (1904). The proof and measurement of association between two things. *The American Journal of Psychology, 15*(1), 72-101.

Kendall, M. G., & Stuart, A. (1973). *The Advanced Theory of Statistics, Vol. 2: Inference and Relationship*. Griffin.
