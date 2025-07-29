# G-Squared Test

The G-Squared test, also known as the G-test or more formally as the likelihood-ratio test for contingency tables, is a conditional independence test for categorical (discrete) data. It is a powerful alternative to the more traditional Pearson's Chi-Squared test and is a standard method used in constraint-based causal discovery algorithms for discrete variables (Spirtes et al., 2000).

The theoretical foundation for the test is Wilks's theorem, which shows that the distribution of the log-likelihood ratio statistic asymptotically approaches a Chi-Square ($\chi^2$) distribution (Wilks, 1938). The G-test is often preferred by statisticians due to its mathematical properties, such as additivity. It is also directly related to information theory, as the G-statistic is proportional to the mutual information between the variables (Cover & Thomas, 2006).

## Mathematical Formulation

The test statistic is calculated from the observed frequencies (O) and the expected frequencies (E) in a contingency table constructed from the data. The expected frequencies are calculated under the null hypothesis of independence. The formula for the G-statistic is:

```{math}
G = 2 \sum_{i} O_i \ln\left(\frac{O_i}{E_i}\right)
```

where the sum is taken over all non-empty cells i in the contingency table. For a conditional independence test of $X \perp Y | Z$, this calculation is performed for each stratum (i.e., for each specific value of the conditioning variable Z), and the resulting G-statistics are summed.

Under the null hypothesis, the total G-statistic is asymptotically distributed as a Chi-Square ($\chi^2$) random variable. The degrees of freedom (df) are calculated as:

```{math}
df = (|X| - 1)(|Y| - 1) \prod_{z \in Z} |z|
```

where $|V|$ denotes the number of distinct categories for a variable V.

## Assumptions

- **Categorical Data**: The variables under consideration must be discrete (categorical).
- **Independent Samples**: The observations are assumed to be drawn independently from the population.
- **Sufficient Sample Size**: As an asymptotic test, its validity depends on the sample size being large enough. While the G-test is often considered more reliable than Pearson's Chi-Squared test for smaller sample sizes (Sokal & Rohlf, 1981), caution is still advised. A common rule of thumb is that the test may be unreliable if more than 20% of the cells in the contingency table have an expected frequency of less than 5.

## Code Example

```python
import numpy as np
from citk.tests import GSq

# Generate discrete data for a chain: X -> Z -> Y
n = 500
X = np.random.randint(0, 3, size=n)
Z = (X + np.random.randint(0, 2, size=n)) % 3
Y = (Z + np.random.randint(0, 2, size=n)) % 3
data = np.vstack([X, Y, Z]).T

# Initialize the test
g_sq_test = GSq(data)

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = g_sq_test(0, 1)
print(f"P-value (unconditional) for X _||_ Y: {p_value_unconditional:.4f}")

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = g_sq_test(0, 1, [2])
print(f"P-value (conditional) for X _||_ Y | Z: {p_value_conditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.simple_tests.GSq`.

## References

Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley-Interscience.

Sokal, R. R., & Rohlf, F. J. (1981). Biometry: The Principles and Practice of Statistics in Biological Research. W. H. Freeman.

Spirtes, P., Glymour, C. N., & Scheines, R. (2000). Causation, prediction, and search. MIT press.

Wilks, S. S. (1938). The large-sample distribution of the likelihood ratio for testing composite hypotheses. The Annals of Mathematical Statistics, 9(1), 60-62. 