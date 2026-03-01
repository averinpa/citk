# `citk`: A Conditional Independence Test Toolkit

`citk` is a Python library that provides a comprehensive and modern toolkit for conditional independence (CI) testing. It is designed to be seamlessly integrated with the [`causal-learn`](https://github.com/cmu-phil/causal-learn) package and offers a collection of classical, statistical, and advanced machine learning-based CI tests.

The library is structured to be a powerful benchmark for causal discovery and a practical toolkit for researchers and practitioners.

## Features

- **Wide Range of Tests**: Includes classical tests (Fisher's Z, Spearman, G-Squared, Chi-Squared), statistical model-based tests (GLM-based), and modern ML-based tests (KCI, Random Forest, DML, CRIT, EDML).
- **`causal-learn` Compatible**: All tests are designed as drop-in replacements for the standard tests in the `causal-learn` ecosystem, allowing you to easily use them with algorithms like PC.

## Installation

Install directly from GitHub with `pip`:

```bash
pip install git+https://github.com/averinpa/citk.git
```

For local development with extras:

```bash
uv sync --all-extras
```

Core CI tests no longer require LightGBM. If you want to run custom LightGBM-based models, install the optional extra:

```bash
uv sync --extra ml
```

R-backed tests are optional and require:
- `rpy2` Python package (`pip install 'citk[r]'` or `uv sync --extra r`)
- R package `RCIT` from GitHub `ericstrobl/RCIT`
- R package `bnlearn` from CRAN

## Quickstart Example

Here is a simple example of how to use a `citk` test within the `causal-learn` PC algorithm.

```python
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
import citk.tests 

# 1. Generate some data
np.random.seed(42)
data = np.random.randn(200, 3)
data[:, 2] = 0.5 * data[:, 0] + 0.5 * data[:, 1] + 0.1 * np.random.randn(200)

# 2. Run the PC algorithm using a citk test
# Example test ids: "fisherz_citk", "spearman", "gsq", "chisq", "rf", "dml"
cg = pc(data, alpha=0.05, indep_test='spearman')

# 3. View the learned graph
print("Learned Graph Edges:")
print(cg.G.get_edges())
```

## Available Tests

| Test Name | Family | Wrapped From |
|---|---|---|
| `fisherz_citk` | Simple | `causal-learn` (`CIT(..., method_name="fisherz")`) |
| `spearman` | Simple | `causal-learn` Fisher-Z on ranked data |
| `gsq` | Simple | `causal-learn` (`Chisq_or_Gsq(..., method_name="gsq")`) |
| `chisq` | Simple | `causal-learn` (`Chisq_or_Gsq(..., method_name="chisq")`) |
| `reg` | Statistical Model | `statsmodels` OLS likelihood-ratio CI |
| `logit` | Statistical Model | `statsmodels` Logit likelihood-ratio CI |
| `pois` | Statistical Model | `statsmodels` Poisson GLM likelihood-ratio CI |
| `kci` | ML-Based | `causal-learn` KCI wrapper |
| `rf` | ML-Based | `scikit-learn` RandomForest + permutation CI |
| `dml` | ML-Based | `scikit-learn` residualization + `statsmodels` residual regression test |
| `crit` | ML-Based | `scikit-learn` quantile models + `statsmodels` residual regression test |
| `edml` | ML-Based | `scikit-learn` residualization + e-value betting |
| `gcm_linear` | GCM | Native `citk` (OLS residualization + asymptotic normal test) |
| `gcm_rf` | GCM | Native `citk` (RandomForest residualization + asymptotic normal test) |
| `wgcm_rf` | GCM | Native `citk` (sample-split weighted GCM with RandomForest) |
| `disc_chisq` | Adapter | Native `citk` equal-frequency discretization + `causal-learn` Chi-Square |
| `disc_gsq` | Adapter | Native `citk` equal-frequency discretization + `causal-learn` G-Square |
| `dummy_fisherz` | Adapter | Native `citk` one-hot encoding + `causal-learn` Fisher-Z aggregation |
| `rcot` | Optional R-Based | R `RCIT::RCoT` via `rpy2` |
| `rcit` | Optional R-Based | R `RCIT::RCIT` via `rpy2` |

For detailed documentation on each test and its parameters, please see our full documentation page [HERE](https://averinpa.github.io/citk/).
