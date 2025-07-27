# How to Choose a Conditional Independence Test

Choosing the right conditional independence (CI) test is crucial for the validity of your causal discovery or feature selection analysis. The appropriate test depends on the characteristics of your data and the underlying assumptions you are willing to make.

## Key Considerations

Here are the primary factors to consider when selecting a test:

### 1. Data Type

- **Continuous Data**: If your variables are all continuous, you have several options:
    - `fisherz`: Assumes linear relationships and multivariate normal data. It is very fast but may fail if these assumptions are violated.
    - `spearman`: A non-parametric alternative that works on ranked data. It is suitable for monotonic (but not necessarily linear) relationships.
    - `kci`: A kernel-based test that can capture complex, non-linear relationships. It is powerful but computationally more intensive.

- **Discrete Data**: If your variables are categorical:
    - `gsq` (G-Square) or `chisq` (Chi-Square): Both are classical tests for discrete data based on contingency tables. `gsq` is often preferred for theoretical reasons, especially with smaller sample sizes.

- **Mixed Data**: When you have a combination of continuous and discrete variables, you currently need to discretize your continuous data to use tests like `gsq` or `chisq`. Future versions may include dedicated tests for mixed data.

### 2. Relationship Type

- **Linear**: If you believe the relationships between your variables are linear, `fisherz` is a computationally efficient choice.
- **Monotonic**: For relationships that are consistently increasing or decreasing but not necessarily linear, `spearman` is a robust option.
- **Non-Linear / Complex**: For arbitrary, complex relationships, machine learning-based tests like `kci` or `rf` are the most powerful and flexible choices, though they come at a higher computational cost.

## Summary Table

| Test Name      | Data Type       | Relationship Type | Key Assumption(s)                                |
|----------------|-----------------|-------------------|--------------------------------------------------|
| `fisherz`      | Continuous      | Linear            | Multivariate normality                           |
| `spearman`     | Continuous      | Monotonic         | Monotonicity                                     |
| `gsq` / `chisq`  | Discrete        | Any               | Adequate sample size for contingency table cells |
| `kci`          | Continuous      | Any               | None (non-parametric)                            |
| `rf` / `dml`   | Continuous      | Any               | None (non-parametric)                            | 