# Fisher's Z Test

The Fisher's Z test is a classical conditional independence test for continuous data. It is most powerful under the assumption of linear relationships and serves as a foundational method in statistical and causal discovery applications. Because of its computational efficiency, it remains the default test used in seminal causal discovery methods like the PC algorithm, even in modern high-dimensional settings (Kalisch & Bühlmann, 2007).

## Mathematical Formulation

The test is based on the sample partial correlation coefficient, $r = \rho(X, Y | Z)$. To create a test statistic with a well-behaved sampling distribution, the test relies on the Fisher's Z-transformation, first developed to handle the "probable error" of correlation coefficients from small samples (Fisher, 1921). The transformation of the partial correlation $r$ is given by:

```{math}
Z(r) = \frac{1}{2} \ln\left(\frac{1+r}{1-r}\right)
```

This can also be expressed as the inverse hyperbolic tangent, `artanh(r)`. This transformation stabilizes the variance and maps the correlation coefficient to an approximately normal distribution, a concept that was later explicitly extended to cover partial correlations (Fisher, 1924). The statistical properties and robustness of this transformation were later explored in extensive detail (Hotelling, 1953).

Under the null hypothesis of conditional independence, the test statistic is constructed as:

```{math}
T = \sqrt{n - |Z| - 3} \cdot |Z(r)|
```

where $n$ is the sample size and $|Z|$ is the number of conditioning variables. This statistic follows a standard normal distribution, $N(0, 1)$, from which the p-value is calculated.

## Assumptions

The validity of the Fisher's Z test rests on two strong assumptions:

- **Multivariate Normality**: All variables (X, Y, and the conditioning set Z) are assumed to be drawn from a multivariate normal distribution. In such a distribution, zero partial correlation is mathematically equivalent to conditional independence, a property that is foundational to constraint-based causal discovery (Spirtes et al., 2000).
- **Linearity**: The relationships between the variables are assumed to be linear. The test may have low power to detect non-linear dependencies.

## Code Example

```python
import numpy as np
from citk.tests import FisherZ

# Generate data where X and Y are independent given Z
# X -> Z -> Y
n = 500
X = np.random.randn(n)
Z = 2 * X + np.random.randn(n)
Y = 3 * Z + np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Initialize the test
fisher_z_test = FisherZ(data)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = fisher_z_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = fisher_z_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.simple_tests.FisherZ`.

## References

Fisher, R. A. (1921). On the 'probable error' of a coefficient of correlation deduced from a small sample. *Metron, 1*(4), 1-32.

Fisher, R. A. (1924). The distribution of the partial correlation coefficient. *Metron, 3*(3-4), 329-332.

Hotelling, H. (1953). New light on the correlation coefficient and its transforms. *Journal of the Royal Statistical Society: Series B (Methodological), 15*(2), 193-225.

Kalisch, M., & Bühlmann, P. (2007). Estimating high-dimensional directed acyclic graphs with the PC-algorithm. *Journal of Machine Learning Research, 8*, 613-636.

Spirtes, P., Glymour, C. N., & Scheines, R. (2000). *Causation, prediction, and search*. MIT press. 