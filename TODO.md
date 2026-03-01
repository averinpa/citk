# citk TODO

## Bugs

- [x] `_glm_conditional_independence_test` mutates `df.columns` in place — fixed with `.copy()`.
- [x] `pingouin` imported but never used — removed.
- [x] `Ridge` imported but never used — removed.
- [x] `pd.api.types.is_categorical_dtype` deprecated — replaced with `dtype.name == 'category'`.
- [x] `pyproject.toml` missing dependencies — added scikit-learn and statsmodels; rpy2 as optional (`lightgbm` moved to optional extra).
- [x] `pyproject.toml` author placeholder — updated to Pavel Averin.

## Architecture improvements

- [x] **Extract cache boilerplate into base class.** `CITKTest.__call__` handles caching, subclasses implement `_compute`.
- [x] **Declare supported data types per test.** `supported_dtypes` class attribute on all tests.

## Remove lightgbm as a core dependency

- [x] **DML/EDML: replace default LightGBM regressor with `sklearn.ensemble.HistGradientBoostingRegressor`** — users can still pass a custom LightGBM model via `model=` kwarg.
- [x] **CRIT: replace hardcoded LightGBM quantile regressors in `_conformalized_ci_test`** — switched to `sklearn.ensemble.GradientBoostingRegressor(loss='quantile', alpha=...)` with optional `quantile_model_factory`.
- [x] **Move lightgbm from `dependencies` to `[project.optional-dependencies]`** in `pyproject.toml` — now in optional `ml` extra.
- [x] **Update smoke tests** — ML smoke checks no longer require lightgbm or dcor.

## New tests for Paper 1 benchmark

### Kernel family (via rpy2 — R RCIT package)

- [x] **RCoT** — Implemented via R `RCIT::RCoT` wrapper and registered as `'rcot'`.
- [x] **RCIT** — Implemented via R `RCIT::RCIT` wrapper and registered as `'rcit'`.
- [ ] **KCI via R** — Current KCI wraps causal-learn which has numerical overflow issues. Rewrite to use R `RCIT::KCIT` via rpy2. Cap at n=2000. Keep registration as `'kci'`.

### kNN CMI family

- [ ] **CMIknn** — Conditional mutual information via kNN density estimation (Runge 2018). Wrap `tigramite.independence_tests.cmiknn.CMIknn`. Register as `'cmiknn'`.
- [ ] **CMIknnMixed** — Mixed-type CMI via kNN for multivariate discrete/categorical/continuous variables (Runge, tigramite). Permutation-based test. Wrap `tigramite.independence_tests.cmiknn.CMIknnMixed`. Register as `'cmiknn_mixed'`.
- [ ] **mCMIkNN** — Mixed-type CMI via kNN (Huegle et al. 2023). Wrap from `/Users/pavelaverin/Projects/mCMIkNN/src`. Register as `'mcmiknn'`.

### Regression family

- [ ] **RegressionCI** — Parametric regression-based CI test for mixed data (Runge, tigramite). Tests X ind Y given Z by comparing nested regressions Y|XZ vs Y|Z using deviance. Handles arbitrary mix of continuous and categorical variables. Wrap `tigramite.independence_tests.regressionCI.RegressionCI`. Register as `'regci'`.

### GCM family

- [x] **GCM-linear** — Implemented with OLS residualization and asymptotic normal test statistic. Registered as `'gcm_linear'`.
- [x] **GCM-RF** — Implemented with Random Forest residualization. Registered as `'gcm_rf'`.
- [x] **WGCM-RF** — Implemented with RF sample splitting and weighted residual product statistic. Registered as `'wgcm_rf'`.

### Discretization-aware

- [ ] **DCT** — Discreteness-aware CI test (Dong et al. 2025). Wrap from `/Users/pavelaverin/Projects/DCT`. Scipy shim already applied. Register as `'dct'`. Secondary test — assumes continuous data was discretized, not natively categorical.

### Adapter strategies

These wrap existing tests with a data transformation step. Each is a thin wrapper, not a new test family.

- [x] **Discretize + Chi-squared** — Equal-frequency discretization adapter implemented. Registered as `'disc_chisq'`.
- [x] **Discretize + G-squared** — Equal-frequency discretization adapter implemented. Registered as `'disc_gsq'`.
- [x] **Dummy-code + Fisher Z** — One-hot encoding adapter with Fisher combined p-value implemented. Registered as `'dummy_fisherz'`.
- [ ] **Hartemink + Chi-squared** — Hartemink information-preserving discretization via R bnlearn, then chi-squared. Register as `'hartemink_chisq'`.

## Remove unused regression-based tests

These tests are not needed for Paper 1 benchmark and will be replaced by RegressionCI from tigramite.

- [ ] **Delete Regression** (`'reg'`) — Remove `Regression` class from `citk/tests/statistical_model_tests.py`.
- [ ] **Delete Logit** (`'logit'`) — Remove `Logit` class from `citk/tests/statistical_model_tests.py`.
- [ ] **Delete Poisson** (`'pois'`) — Remove `Poisson` class from `citk/tests/statistical_model_tests.py`.
- [ ] **Clean up registrations and imports** — Remove deleted tests from `__init__.py`, update `TEST_REGISTRY`, remove unused smoke tests.

## rpy2 integration

- [x] **Make rpy2 optional.** Current R-backed tests use a lazy import and raise clear install guidance when `rpy2`/`RCIT` is missing.
- [x] **Create `citk/tests/r_based_tests.py`** — rpy2-dependent tests isolated and imported conditionally in `__init__.py`.
- [x] **Document R package requirements** — RCIT package from GitHub (`ericstrobl/RCIT`), bnlearn from CRAN.

## Testing

- [x] **Add pytest suite.** Smoke tests for all 12 existing tests (null → p > 0.05, dependent → p < 0.05).
- [x] **CI via GitHub Actions** — run pytest on push. rpy2 tests can be skipped in CI if R is not available.

## Switch to uv

- [x] **Replace setuptools with hatchling** in `pyproject.toml` build-system.
- [x] **Fix `pyproject.toml`** — author, requires-python, dependencies.
- [x] **Add optional dependency groups** — r, docs, dev.
- [x] **Add `.python-version`** file with `3.11`.
- [x] **Add `uv.lock` to `.gitignore`.**
- [x] **Delete `environment.yml`.**
- [x] **Run `uv sync --all-extras`** to create venv.

## Documentation

- [x] Update README with new test inventory.
- [x] Update `docs/source/guides/choosing_a_test.md` with new tests.
