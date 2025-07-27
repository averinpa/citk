# A Taxonomy of Conditional Independence Tests

The conditional independence (CI) tests provided in `citk` can be broadly classified into several categories based on their underlying principles. Understanding these categories can help you reason about which test might be most appropriate for your research question and data.

## 1. Correlation-Based Tests

These tests are based on measures of correlation, typically Pearson correlation for linear relationships or Spearman correlation for monotonic relationships.

- **Core Idea**: Test if the partial correlation between two variables is zero after controlling for the conditioning set.
- **Example**: `fisherz`, `spearman`
- **Strengths**: Computationally very fast and statistically efficient if the assumptions (e.g., linearity, normality) hold.
- **Weaknesses**: Can fail to detect non-linear or complex dependencies.

## 2. Contingency Table-Based Tests

These are classical statistical tests designed for discrete (categorical) data.

- **Core Idea**: Compare the observed cell counts in a contingency table to the counts that would be expected under the null hypothesis of independence.
- **Example**: `gsq` (G-test), `chisq` (Chi-Square)
- **Strengths**: Well-understood statistical properties and robust for categorical data.
- **Weaknesses**: Requires discrete data. Can suffer from low statistical power if the sample size is small relative to the number of cells in the table.

## 3. Regression-Based Tests

These tests use regression models to check for independence.

- **Core Idea**: Regress one variable onto the other variables (including the conditioning set). If the coefficient for the target variable is statistically indistinguishable from zero, it suggests independence.
- **Example**: `reg` (Linear Regression), `logit` (Logistic Regression), `pois` (Poisson Regression)
- **Strengths**: Can be tailored to the specific data generating process (e.g., binary outcomes with `logit`).
- **Weaknesses**: Make strong parametric assumptions about the functional form of the relationship.

## 4. Kernel-Based Tests

These are non-parametric tests that operate in a high-dimensional feature space defined by a kernel function.

- **Core Idea**: Map the data into a Reproducing Kernel Hilbert Space (RKHS) and test for independence in that space. This allows the detection of complex, non-linear relationships. The Hilbert-Schmidt Independence Criterion (HSIC) is a common measure used.
- **Example**: `kci`
- **Strengths**: Can detect any kind of relationship (linear, non-linear, non-monotonic). Does not make strong assumptions about the data distribution.
- **Weaknesses**: Computationally more expensive than simpler tests. The choice of kernel and its parameters can influence the results.

## 5. Machine Learning-Based Tests

This is a modern and flexible category of tests that leverage machine learning models to test for independence.

- **Core Idea**: Use a predictive model (like a random forest or gradient boosting) to see if one variable can be predicted from another, given the conditioning set. If the prediction is no better than random chance, it implies independence.
- **Example**: `rf` (Random Forest), `dml` (Double Machine Learning)
- **Strengths**: Highly flexible, can capture very complex relationships, and often make few assumptions.
- **Weaknesses**: Can be computationally very intensive. May require larger sample sizes to work effectively. 