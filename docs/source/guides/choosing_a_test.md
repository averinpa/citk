# How to Choose a Conditional Independence Test

Choosing the right conditional independence (CI) test is crucial for the validity of your causal discovery or feature selection analysis. The appropriate test depends on the characteristics of your data and the underlying assumptions you are willing to make.

## Key Considerations

Here are the primary factors to consider when selecting a test:

### 1. Data Type

- **Continuous Data**: If your variables are all continuous, you have several options:
    - `fisherz_citk`: Assumes linear relationships and multivariate normal data. It is very fast but may fail if these assumptions are violated.
    - `spearman`: A non-parametric alternative that works on ranked data. It is suitable for monotonic (but not necessarily linear) relationships.
    - `kci`: Optional R-backed KCIT (`RCIT::KCIT`) for complex, non-linear relationships.
    - `rcot`, `rcit`: Optional R-backed kernel/random-feature tests from the RCIT package.
    - `cmiknn`, `cmiknn_mixed`: Optional tigramite kNN-CMI tests.
    - `mcmiknn`: Optional mixed-type kNN CMI wrapper from local mCMIkNN repo.
    - `regci`: Optional tigramite parametric mixed-data regression CI.
    - `rf`, `dml`, `crit`, `edml`: Flexible ML-based options for non-linear structure.

- **Discrete Data**: If your variables are categorical:
    - `gsq` (G-Square) or `chisq` (Chi-Square): Classical tests based on contingency tables.

- **Mixed Data**: Current built-in options are limited; discretization is still a practical baseline for `gsq`/`chisq`.
    - `disc_chisq`, `disc_gsq`: Equal-frequency discretization adapters around classical discrete tests.
    - `dummy_fisherz`: One-hot encoding adapter with Fisher-Z aggregation.
    - `hartemink_chisq`: Information-preserving Hartemink discretization (via R `bnlearn`) + Chi-square.
    - `dct`: Optional DCT wrapper from local DCT repository.

### 2. Relationship Type

- **Linear**: If you believe the relationships between your variables are linear, `fisherz` is a computationally efficient choice.
- **Monotonic**: For relationships that are consistently increasing or decreasing but not necessarily linear, `spearman` is a robust option.
- **Non-Linear / Complex**: For arbitrary, complex relationships, machine learning-based tests like `kci` or `rf` are the most powerful and flexible choices, though they come at a higher computational cost.

## Summary Table

| Test Name | Data Type | Relationship Type | Key Assumption(s) |
|-----------|-----------|-------------------|-------------------|
| `fisherz_citk` | Continuous | Linear | Approximate Gaussianity |
| `spearman` | Continuous | Monotonic | Monotonicity |
| `gsq` / `chisq` | Discrete | Any | Adequate contingency support |
| `kci` | Continuous | Any | Requires `rpy2` + R `RCIT` package |
| `rcot` / `rcit` | Continuous | Any | Requires `rpy2` + R `RCIT` package |
| `cmiknn` / `cmiknn_mixed` / `regci` | Mixed or continuous | Any | Requires `tigramite` |
| `mcmiknn` | Mixed | Any | Requires local mCMIkNN repository |
| `rf` / `dml` / `crit` / `edml` | Continuous | Any | ML residualization quality |
| `gcm_linear` / `gcm_rf` / `wgcm_rf` | Continuous | Any | Residual covariance test |
| `disc_chisq` / `disc_gsq` | Mixed or continuous | Any | Discretization quality |
| `dummy_fisherz` | Mixed or discrete | Any | One-hot encoding fidelity |
| `hartemink_chisq` | Mixed or continuous | Any | Requires `rpy2` + R `bnlearn` |
| `dct` | Mixed or discretized continuous | Any | Requires local DCT repository |
