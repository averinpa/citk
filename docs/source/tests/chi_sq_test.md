# Chi-Squared Test

The Chi-Squared ($\chi^2$) test is a classical statistical test for categorical (discrete) data. Developed by Karl Pearson at the turn of the 20th century, it was one of the first "goodness-of-fit" tests, designed to assess whether an observed set of frequencies differs from a theoretical distribution (Pearson, 1900). In the context of contingency tables, it is used to test for the conditional independence of two variables, X and Y, given a set of variables, Z.

## Mathematical Formulation

The test compares the observed frequencies (O) in a contingency table with the frequencies that would be expected (E) if the null hypothesis of independence were true. The Pearson Chi-Squared statistic is calculated as:

```{math}
\chi^2 = \sum_{i} \frac{(O_i - E_i)^2}{E_i}
```

where the sum is over all cells i in the contingency table. This formula can be seen as a second-order Taylor approximation to the log-likelihood ratio (G-test) statistic, to which it is asymptotically equivalent.

Under the null hypothesis, this statistic follows a Chi-Squared ($\chi^2$) distribution with degrees of freedom given by:

```{math}
df = (|X| - 1)(|Y| - 1) \prod_{z \in Z} |z|
```

where $|V|$ denotes the number of distinct categories for a variable V.

## Assumptions

- **Categorical Data**: The data must be categorical (discrete).
- **Independent Observations**: The individual observations must be independent of each other.
- **Sufficient Sample Size**: The sample size should be large enough that the expected frequency in each cell is not too small. A widely cited rule of thumb, often attributed to Cochran, suggests that the test may be inappropriate if more than 20% of the cells have an expected frequency below 5, or if any cell has an expected frequency below 1 (Cochran, 1954).

## Code Example

```python
import numpy as np
from citk.tests import ChiSq

# Generate discrete data representing a collider: X -> Y <- Z
n = 500
X = np.random.randint(0, 2, size=n)
Z = np.random.randint(0, 2, size=n)
Y = (X + Z + np.random.randint(0, 2, size=n)) % 2
data = np.vstack([X, Y, Z]).T

# Initialize the test
chisq_test = ChiSq(data)

# Test for unconditional independence (X and Z are independent)
p_value_unconditional = chisq_test(0, 2)
print(f"P-value for X _||_ Z: {p_value_unconditional:.4f}")

# Test for conditional dependence on the collider Y
p_value_conditional = chisq_test(0, 2, [1])
print(f"P-value for X _||_ Z | Y: {p_value_conditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.simple_tests.ChiSq`.

## References

Cochran, W. G. (1954). Some methods for strengthening the common $\chi^2$ tests. Biometrics, 10(4), 417-451.

Pearson, K. (1900). On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling. Philosophical Magazine, 50(302), 157-175. 