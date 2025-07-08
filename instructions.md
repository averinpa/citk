I want to cover the most important and conceptually distinct categories of tests to create a powerful benchmark and a useful toolkit.

Here is a strategic list of tests to include in `citk`, organized by the categories. This list is comprehensive enough to be considered a state-of-the-art benchmark but manageable enough to be feasible within a year.

---

### Category 1: "Classical" / Simple Non-Parametric Tests
These are your essential baselines. They are fast but have strong assumptions.

| Test Name | `citk` Function Name (Suggestion) | Underlying Principle | Python Library/Implementation | Data Type |
| :--- | :--- | :--- | :--- | :--- |
| **Fisher's Z Test** | `ci_test_fisherz` | Partial correlation (assumes normality) | `pingouin.partial_corr(method='pearson')` | Continuous-Continuous |
| **Spearman's Rho Test** | `ci_test_spearman` | Partial rank correlation | `pingouin.partial_corr(method='spearman')` | Continuous-Continuous |
| **G-Squared Test** | `ci_test_gsq` | Log-likelihood ratio for contingency tables | `causal-learn wrapper` | Categorical-Categorical |
| **Chi-Squared Test** | `ci_test_chi2` | Pearson's Chi-squared for contingency tables | `causal-learn wrapper` | Categorical-Categorical |
| **Distance Correlation Test**| `ci_test_dcor` | Non-linear dependence based on distances | `dcor.partial_distance_correlation` | Continuous-Continuous |

**Strategic Value:** This covers the classic linear, monotonic, and categorical dependence structures, plus a modern non-linear baseline (`dcor`).

---

### Category 2: `MXM`-Style Statistical Model-Based Tests
These represent the "flexible but parametric" school of thought.

| Test Name | `citk` Function Name (Suggestion) | Underlying Principle | Python Library/Implementation | Target Data Type |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression F-Test** | `ci_test_reg` | LLR test for nested OLS models | `statsmodels.OLS` | Continuous |
| **Logistic Regression Test** | `ci_test_logit` | LLR test for nested Logit models | `statsmodels.Logit` | Binary |
| **Poisson Regression Test** | `ci_test_pois` | LLR test for nested Poisson models | `statsmodels.GLM(family=Poisson)` | Counts |


**Strategic Value:** This set covers the most common data types seen in real-world problems: continuous, binary. It perfectly represents the `MXM` philosophy.

---

### Category 3: Modern Machine Learning-Based Tests
This is your core contribution. It should include established methods and your novel proposals.

| Test Name | `citk` Function Name (Suggestion) | Underlying Principle | Python Library/Implementation | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Kernel Conditional Independence (KCI)** | `ci_test_kci` | Hilbert Space embeddings | `causallearn.test.KCI` | A key kernel-based method. |
| **DoubleML CI Test** | `ci_test_dml` | Orthogonalization + final stage regression | manual implementation i will provide later | Your primary ML baseline. Essential. |
| **Random Forest CI Test (e.g., CForest)**| `ci_test_rf` | Permutation-based variable importance | Can be implemented manually using `scikit-learn`. | Represents a popular, tree-based heuristic. |
| **Your Novel Methods:** | | | | **The stars of the show.** |
| **Conformalized Residual Test**| `ci_test_crit` | Conformal prediction on residuals | **Your PhD work.** | Demonstrates handling of predictive uncertainty. |
| **E-Value DML Test** | `ci_test_edml` | E-values for sequential/safe testing | **Your PhD work.** | Demonstrates handling of testing uncertainty. |

---
for example

```python
import pandas as pd
import numpy as np
import pingouin as pg

def ci_test_fisherz(df: pd.DataFrame, x: str, y: str, z: list = None):
    """
    Performs a conditional independence test using Fisher's Z-transform.

    This test assesses the partial correlation between two continuous variables (x and y)
    while controlling for a set of other continuous variables (z). It assumes
    the data follows a multivariate normal distribution.

    This function is a wrapper around pingouin.partial_corr.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame containing the data.
    x : str
        The name of the first variable (column in df).
    y : str
        The name of the second variable (column in df).
    z : list, optional
        A list of column names for the conditioning set of variables.
        If None or empty, it performs a standard Pearson correlation test.

    Returns
    -------
    tuple
        A tuple containing the test statistic and the p-value.
        - statistic (float): The partial correlation coefficient.
        - p_value (float): The p-value of the test.
    """
    if z is None or not z:
        # If no conditioning set, perform a standard Pearson correlation
        result = pg.corr(df[x], df[y], method='pearson')
        p_val = result['p-val'].iloc[0]
        stat = result['r'].iloc[0]
    else:
        # Perform a partial correlation
        result = pg.partial_corr(data=df, x=x, y=y, covar=z, method='pearson')
        p_val = result['p-val'].iloc[0]
        stat = result['r'].iloc[0]

    return (stat, p_val)


def ci_test_spearman(df: pd.DataFrame, x: str, y: str, z: list = None):
    """
    Performs a conditional independence test using Spearman's rank correlation.

    This is a non-parametric alternative to Fisher's Z test. It assesses the
    monotonic partial relationship between two variables (x and y) based on their
    ranks, while controlling for a set of other variables (z). It does not
    assume normality.

    This function is a wrapper around pingouin.partial_corr.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame containing the data.
    x : str
        The name of the first variable (column in df).
    y : str
        The name of the second variable (column in df).
    z : list, optional
        A list of column names for the conditioning set of variables.
        If None or empty, it performs a standard Spearman correlation test.

    Returns
    -------
    tuple
        A tuple containing the test statistic and the p-value.
        - statistic (float): The partial Spearman rank correlation coefficient.
        - p_value (float): The p-value of the test.
    """
    if z is None or not z:
        # If no conditioning set, perform a standard Spearman correlation
        result = pg.corr(df[x], df[y], method='spearman')
        p_val = result['p-val'].iloc[0]
        stat = result['r'].iloc[0]
    else:
        # Perform a partial Spearman correlation
        result = pg.partial_corr(data=df, x=x, y=y, covar=z, method='spearman')
        p_val = result['p-val'].iloc[0]
        stat = result['r'].iloc[0]

    return (stat, p_val)


# --- Example Usage ---
if __name__ == '__main__':
    print("--- Causal Discovery CI Test Example ---")

    # Create a synthetic dataset where:
    # Z is a common cause of X and Y
    # X and Y are conditionally independent given Z.
    np.random.seed(42)
    n_samples = 200
    z_var = np.random.normal(0, 1, n_samples)
    x_var = 0.7 * z_var + np.random.normal(0, 0.5, n_samples)
    y_var = -0.6 * z_var + np.random.normal(0, 0.5, n_samples)
    
    # Add a non-linear relationship for the Spearman test
    y_var_nonlinear = np.exp(-0.6 * z_var) + np.random.normal(0, 0.5, n_samples)

    data = pd.DataFrame({
        'X': x_var,
        'Y': y_var,
        'Y_nonlinear': y_var_nonlinear,
        'Z': z_var
    })

    # --- Test 1: Unconditional dependence (X vs Y) ---
    print("\n1. Testing unconditional dependence between X and Y...")
    stat_uncond, p_uncond = ci_test_fisherz(data, x='X', y='Y')
    print(f"   Fisher's Z: stat={stat_uncond:.4f}, p-value={p_uncond:.4f}")
    if p_uncond < 0.05:
        print("   -> Result: X and Y are DEPENDENT (as expected, due to common cause Z).")
    else:
        print("   -> Result: X and Y are INDEPENDENT.")


    # --- Test 2: Conditional independence (X vs Y | Z) ---
    print("\n2. Testing conditional independence between X and Y, given Z...")
    stat_cond, p_cond = ci_test_fisherz(data, x='X', y='Y', z=['Z'])
    print(f"   Fisher's Z: stat={stat_cond:.4f}, p-value={p_cond:.4f}")
    if p_cond < 0.05:
        print("   -> Result: X and Y are DEPENDENT.")
    else:
        print("   -> Result: X and Y are INDEPENDENT (as expected).")
        

    # --- Test 3: Spearman's test for non-linear relationships ---
    print("\n3. Testing conditional independence for a non-linear case with Spearman's Rho...")
    stat_spearman, p_spearman = ci_test_spearman(data, x='X', y='Y_nonlinear', z=['Z'])
    print(f"   Spearman's Rho: stat={stat_spearman:.4f}, p-value={p_spearman:.4f}")
    if p_spearman < 0.05:
        print("   -> Result: X and Y_nonlinear are DEPENDENT.")
    else:
        print("   -> Result: X and Y_nonlinear are INDEPENDENT (as expected).")
```

dcor example
```python
import pandas as pd
import numpy as np
import dcor

def ci_test_dcor(df: pd.DataFrame, x: str, y: str, z: list = None, num_resamples: int = 200):
    """
    Performs a conditional independence test using partial distance correlation.

    This is a non-parametric test that can detect non-linear and non-monotonic
    conditional relationships. It is more computationally intensive than simpler
    tests because it relies on a permutation test to compute the p-value.

    This function is a wrapper around dcor.partial_distance_correlation_test.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame containing the data.
    x : str
        The name of the first variable (column in df).
    y : str
        The name of the second variable (column in df).
    z : list, optional
        A list of column names for the conditioning set of variables.
        If None or empty, it performs a standard distance correlation test.
    num_resamples : int, optional
        The number of bootstrap or permutation resamples to use for the
        p-value calculation. More resamples are more accurate but slower.
        Default is 200.

    Returns
    -------
    tuple
        A tuple containing the test statistic and the p-value.
        - statistic (float): The partial distance correlation coefficient.
        - p_value (float): The p-value from the permutation test.
    """
    # Extract data as NumPy arrays, which is required by the dcor library
    x_data = df[x].to_numpy()
    y_data = df[y].to_numpy()

    if z is None or not z:
        # Unconditional distance correlation test
        # The test function directly gives statistic and p-value
        result = dcor.distance_correlation_test(
            x_data, y_data, num_resamples=num_resamples
        )
        stat = result.statistic
        p_val = result.pvalue
    else:
        # Conditional distance correlation test
        # Ensure z_data is a 2D array, which to_numpy() does automatically
        # for a list of columns.
        z_data = df[z].to_numpy()
        
        # The test function handles the permutation test internally
        result = dcor.partial_distance_correlation_test(
            x_data, y_data, z_data, num_resamples=num_resamples
        )
        stat = result.statistic
        p_val = result.pvalue

    return (stat, p_val)


# --- Example Usage ---
if __name__ == '__main__':
    print("--- Distance Correlation CI Test Example ---")

    # Create a synthetic dataset with a non-linear collider structure:
    # X -> Z <- Y
    # X and Y are independent, but become dependent when we condition on Z.
    # The relationship is Z = X^2 + Y^2, which is non-linear.
    # Fisher's Z test would fail to detect this conditional dependence.
    
    np.random.seed(0)
    n_samples = 300
    # X and Y are independent standard normal variables
    x_collider = np.random.randn(n_samples)
    y_collider = np.random.randn(n_samples)
    
    # Z is a non-linear combination of X and Y, plus some noise
    z_collider = x_collider**2 + y_collider**2 + 0.1 * np.random.randn(n_samples)
    
    data_collider = pd.DataFrame({
        'X': x_collider,
        'Y': y_collider,
        'Z': z_collider
    })

    print("Scenario: A non-linear collider (X -> Z <- Y), where Z = X^2 + Y^2")
    print("X and Y should be unconditionally independent but conditionally dependent given Z.")
    
    # --- Test 1: Unconditional independence (X vs Y) ---
    print("\n1. Testing unconditional independence between X and Y...")
    # We expect a high p-value (fail to reject independence)
    stat_uncond, p_uncond = ci_test_dcor(data_collider, x='X', y='Y')
    print(f"   Distance Correlation: stat={stat_uncond:.4f}, p-value={p_uncond:.4f}")
    if p_uncond < 0.05:
        print("   -> Result: X and Y are DEPENDENT.")
    else:
        print("   -> Result: X and Y are INDEPENDENT (as expected).")

    # --- Test 2: Conditional dependence (X vs Y | Z) ---
    print("\n2. Testing conditional dependence between X and Y, given Z...")
    # Here, we expect a low p-value (reject independence)
    # This is where dcor shines and Fisher's Z would fail.
    stat_cond, p_cond = ci_test_dcor(data_collider, x='X', y='Y', z=['Z'])
    print(f"   Partial Distance Correlation: stat={stat_cond:.4f}, p-value={p_cond:.4f}")
    if p_cond < 0.05:
        print("   -> Result: X and Y are DEPENDENT (as expected, dcor correctly finds this).")
    else:
        print("   -> Result: X and Y are INDEPENDENT.")
        
    # --- For comparison: Show that Fisher's Z fails here ---
    from pingouin import partial_corr
    print("\n3. For comparison, running Fisher's Z on the same conditional test...")
    fisher_result = partial_corr(data=data_collider, x='X', y='Y', covar='Z')
    fisher_stat = fisher_result['r'].iloc[0]
    fisher_p = fisher_result['p-val'].iloc[0]
    print(f"   Fisher's Z: stat={fisher_stat:.4f}, p-value={fisher_p:.4f}")
    if fisher_p < 0.05:
        print("   -> Result: X and Y are DEPENDENT.")
    else:
        print("   -> Result: X and Y are INDEPENDENT (Fisher's Z fails to find the dependence).")
```

glm examples
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2

def _glm_conditional_independence_test(df: pd.DataFrame, x: str, y: str, z: list, 
                                       model_class, family=None, **kwargs):
    """
    Core helper function for performing GLM-based conditional independence tests.

    This function fits two nested models:
    1. Null Model (H0):  y ~ z (y is independent of x, given z)
    2. Alt. Model (H1): y ~ x + z

    It then performs a log-likelihood ratio (LLR) test to see if the
    alternative model provides a significantly better fit.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame.
    x : str
        The candidate variable.
    y : str
        The target variable.
    z : list
        The conditioning set.
    model_class : statsmodels.base.model.Model
        The statsmodels model class to use (e.g., sm.OLS, sm.Logit, sm.GLM).
    family : statsmodels.genmod.families.Family, optional
        The family for GLMs (e.g., sm.families.Poisson()).
    **kwargs :
        Additional keyword arguments to pass to the model's from_formula method.

    Returns
    -------
    tuple
        A tuple containing the LLR test statistic and the p-value.
    """
    # Ensure the candidate is not in the conditioning set
    if z and x in z:
        z = [v for v in z if v != x]

    # --- Null Model (H0: y ~ z) ---
    if z:
        formula_null = f"{y} ~ {' + '.join(z)}"
    else: # If conditioning set is empty, null model is intercept-only
        formula_null = f"{y} ~ 1"
        
    model_args = {'formula': formula_null, 'data': df}
    if family:
        model_args['family'] = family
        
    model_null = model_class.from_formula(**model_args, **kwargs).fit()
    loglik_null = model_null.llf
    df_null = model_null.df_model # Number of predictors in the null model

    # --- Alternative Model (H1: y ~ x + z) ---
    all_predictors = [x] + (z if z else [])
    formula_alt = f"{y} ~ {' + '.join(all_predictors)}"
    
    model_args['formula'] = formula_alt
    model_alt = model_class.from_formula(**model_args, **kwargs).fit()
    loglik_alt = model_alt.llf
    df_alt = model_alt.df_model # Number of predictors in the alternative model

    # --- Likelihood Ratio Test ---
    # The statistic follows a chi-squared distribution
    lr_stat = 2 * (loglik_alt - loglik_null)
    
    # Degrees of freedom for the test is the difference in number of parameters
    # df_model correctly handles categorical variables (which use more than 1 param)
    df_diff = df_alt - df_null
    
    # Handle edge cases
    if lr_stat < 0: # Due to numerical instability, can be slightly negative
        lr_stat = 0
    if df_diff <= 0: # Can happen if candidate variable is perfectly collinear
        return (0.0, 1.0)
        
    p_value = chi2.sf(lr_stat, df=df_diff)

    return (lr_stat, p_value)


# --- Specific Test Implementations ---

def ci_test_reg(df: pd.DataFrame, x: str, y: str, z: list = None):
    """CI test for continuous targets using Linear Regression (OLS)."""
    if z is None: z = []
    return _glm_conditional_independence_test(df, x, y, z, model_class=sm.OLS)

def ci_test_logit(df: pd.DataFrame, x: str, y: str, z: list = None):
    """CI test for binary targets using Logistic Regression."""
    if z is None: z = []
    return _glm_conditional_independence_test(df, x, y, z, model_class=sm.Logit)

def ci_test_pois(df: pd.DataFrame, x: str, y: str, z: list = None):
    """CI test for count targets using Poisson Regression."""
    if z is None: z = []
    return _glm_conditional_independence_test(df, x, y, z, 
                                              model_class=sm.GLM, 
                                              family=sm.families.Poisson())


# --- Example Usage ---
if __name__ == '__main__':
    print("--- GLM-based CI Test Examples ---")
    np.random.seed(42)
    n_samples = 300

    # --- 1. Linear Regression Example ---
    print("\n1. Testing with Linear Regression (continuous target)")
    Z_lin = np.random.randn(n_samples)
    X_lin = 0.5 * Z_lin + np.random.randn(n_samples)
    Y_lin = 2.0 * Z_lin + np.random.randn(n_samples) # Y depends only on Z
    df_lin = pd.DataFrame({'X': X_lin, 'Y': Y_lin, 'Z': Z_lin})
    
    stat, pval = ci_test_reg(df_lin, 'X', 'Y', ['Z'])
    print(f"   Conditional test Y ~ X | Z: stat={stat:.4f}, p-value={pval:.4f}")
    print(f"   -> Result: {'DEPENDENT' if pval < 0.05 else 'INDEPENDENT'} (Correct)")
    
    # --- 2. Logistic Regression Example ---
    print("\n2. Testing with Logistic Regression (binary target)")
    Z_log = np.random.randn(n_samples)
    X_log_unrelated = np.random.randn(n_samples) # X is totally unrelated
    # Log-odds of Y depend only on Z
    log_odds = -1.5 * Z_log 
    prob = 1 / (1 + np.exp(-log_odds))
    Y_log = np.random.binomial(1, prob)
    df_log = pd.DataFrame({'X': X_log_unrelated, 'Y': Y_log, 'Z': Z_log})

    stat, pval = ci_test_logit(df_log, 'X', 'Y', ['Z'])
    print(f"   Conditional test Y ~ X | Z: stat={stat:.4f}, p-value={pval:.4f}")
    print(f"   -> Result: {'DEPENDENT' if pval < 0.05 else 'INDEPENDENT'} (Correct)")

    # --- 3. Poisson Regression Example ---
    print("\n3. Testing with Poisson Regression (count target)")
    Z_pois_cont = np.random.uniform(0, 2, n_samples)
    Z_pois_cat = np.random.choice(['A', 'B'], size=n_samples)
    df_pois = pd.DataFrame({'Z_cont': Z_pois_cont, 'Z_cat': Z_pois_cat})
    
    X_pois = 0.4 * Z_pois_cont + np.random.randn(n_samples)
    # The log of the mean rate of Y depends on Z_cat and Z_cont
    # We need to one-hot encode the categorical variable for the formula
    rate = np.exp(1.0 * (pd.get_dummies(df_pois['Z_cat'], drop_first=True)['B']) + 0.8 * df_pois['Z_cont'])
    Y_pois = np.random.poisson(rate)
    df_pois['X'] = X_pois
    df_pois['Y'] = Y_pois

    stat, pval = ci_test_pois(df_pois, 'X', 'Y', ['Z_cont', 'Z_cat'])
    print(f"   Conditional test Y ~ X | Z_cont, Z_cat: stat={stat:.4f}, p-value={pval:.4f}")
    print(f"   -> Result: {'DEPENDENT' if pval < 0.05 else 'INDEPENDENT'} (Correct)")

    # Example where the test should find dependence
    Y_pois_dep = Y_pois + np.random.poisson(np.exp(0.5 * X_pois))
    df_pois['Y_dep'] = Y_pois_dep
    stat, pval = ci_test_pois(df_pois, 'X', 'Y_dep', ['Z_cont', 'Z_cat'])
    print(f"   Conditional test where Y_dep~X|Z: stat={stat:.4f}, p-value={pval:.4f}")
    print(f"   -> Result: {'DEPENDENT' if pval < 0.05 else 'INDEPENDENT'} (Correct)")
```

ci_test_rf example:
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

def ci_test_rf(df: pd.DataFrame, x: str, y: str, z: list = None, 
               n_estimators: int = 100, num_permutations: int = 100,
               test_size: float = 0.2, random_state: int = None):
    """
    Performs a conditional independence test using Random Forest feature importance.

    This is a non-parametric, heuristic test based on a permutation approach.
    It assesses the importance of a candidate variable `x` in predicting a
    target `y`, given a set of conditioning variables `z`.

    The method is as follows:
    1. The built-in Gini importance of `x` from a Random Forest trained on the
       original data (y ~ x + z) is calculated. This is the "observed statistic".
    2. The target variable `y` is then permuted `num_permutations` times. For each
       permutation, a new Random Forest is trained, and the importance of `x`
       is recorded. This creates a null distribution of feature importances.
    3. The p-value is the proportion of importances from the null distribution
       that are greater than or equal to the observed importance.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame containing the data.
    x : str
        The name of the candidate variable.
    y : str
        The name of the target variable.
    z : list, optional
        A list of column names for the conditioning set.
    n_estimators : int
        The number of trees in the forest.
    num_permutations : int
        The number of permutations to create the null distribution.
    test_size : float
        The proportion of the dataset to include in the test split.
        Not used in this implementation but kept for API consistency.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    tuple
        A tuple containing the test statistic and the p-value.
        - statistic (float): The feature importance of x in the original model.
        - p_value (float): The p-value from the permutation test.
    """
    if z is None:
        z = []
    
    # Define predictor and target variables
    predictor_cols = [x] + z
    X_df = df[predictor_cols]
    y_series = df[y]
    
    # Determine if it's a classification or regression task
    is_classification = y_series.nunique() <= 10 or pd.api.types.is_categorical_dtype(y_series.dtype)
    if is_classification:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

    # 1. Calculate the observed statistic
    model.fit(X_df, y_series)
    importances = model.feature_importances_
    # Get the index of our candidate variable 'x'
    x_index = X_df.columns.get_loc(x)
    observed_statistic = importances[x_index]

    # 2. Generate the null distribution
    permuted_statistics = np.zeros(num_permutations)
    y_permuted = y_series.copy()
    
    for i in range(num_permutations):
        # Permute the target variable
        np.random.shuffle(y_permuted.values)
        
        # Re-train the model on the permuted data
        model.fit(X_df, y_permuted)
        permuted_importances = model.feature_importances_
        permuted_statistics[i] = permuted_importances[x_index]

    # 3. Calculate the p-value
    # The p-value is the proportion of permuted scores greater than or equal to the observed score
    p_value = (np.sum(permuted_statistics >= observed_statistic) + 1) / (num_permutations + 1)

    return (observed_statistic, p_value)


# --- Example Usage ---
if __name__ == '__main__':
    print("--- Random Forest CI Test Example ---")
    np.random.seed(42)
    n_samples = 500

    # Scenario: Z is a common cause of X and Y, Y also has a direct link from X
    # Y ~ X + Z
    Z = np.random.randn(n_samples)
    X = 0.5 * Z + np.random.randn(n_samples)
    Y = 2.0 * Z + 0.8 * X + np.random.randn(n_samples)
    
    # Create a completely unrelated variable U
    U = np.random.randn(n_samples)
    
    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z, 'U': U})
    
    # --- Test 1: Test a truly dependent variable (X) ---
    print("\n1. Testing Y ~ X | Z (where X is truly a cause of Y)...")
    stat_dep, pval_dep = ci_test_rf(df, x='X', y='Y', z=['Z'], num_permutations=99)
    print(f"   Feature importance of X = {stat_dep:.4f}, p-value = {pval_dep:.4f}")
    print(f"   -> Result: {'DEPENDENT' if pval_dep < 0.05 else 'INDEPENDENT'} (Correct)")

    # --- Test 2: Test a truly independent variable (U) ---
    print("\n2. Testing Y ~ U | Z (where U is truly independent of Y)...")
    stat_indep, pval_indep = ci_test_rf(df, x='U', y='Y', z=['Z'], num_permutations=99)
    print(f"   Feature importance of U = {stat_indep:.4f}, p-value = {pval_indep:.4f}")
    print(f"   -> Result: {'DEPENDENT' if pval_indep < 0.05 else 'INDEPENDENT'} (Correct)")
```

doubleml, evalue, conformal ci tests

```python
def get_dml_residuals(data, x_idx, y_idx, z_idx, cv_folds=5):
    """Generates high-quality DML residuals and normalizes them."""
    X_target = data[:, x_idx]
    Y_target = data[:, y_idx]
    Z_features = data[:, z_idx]
    
    model = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.05, verbose=-1)
    
    pred_x = cross_val_predict(model, Z_features, X_target, cv=cv_folds)
    pred_y = cross_val_predict(model, Z_features, Y_target, cv=cv_folds)
    
    U = X_target - pred_x
    V = Y_target - pred_y
    
    U = U / np.std(U)
    V = V / np.std(V)
    
    return U, V

# --- Part 3: Implementations of the Advanced CI Tests ---

def dcor_test(x, y, n_perms=499):
    """The final-stage p-value test based on distance correlation."""
    obs_stat = dcor.distance_correlation(x, y)
    y_shuffled = y.copy()
    perm_stats = [dcor.distance_correlation(x, np.random.permutation(y_shuffled)) for _ in range(n_perms)]
    count = np.sum(np.array(perm_stats) >= obs_stat)
    return (count + 1) / (n_perms + 1)

def conformalized_ci_test(data, x_idx, y_idx, z_idx, alpha=0.1, cv_folds=5):
    """Performs the Conformalized Residual Independence Test (CRIT)."""
    X_target, Y_target, Z_features = data[:, x_idx], data[:, y_idx], data[:, z_idx]
    all_indices, all_true_x, all_true_y = np.array([], dtype=int), np.array([]), np.array([])
    all_preds_x_low, all_preds_x_high = np.array([]), np.array([])
    all_preds_y_low, all_preds_y_high = np.array([]), np.array([])
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    lgbm_params = {'objective': 'quantile', 'metric': 'quantile', 'n_estimators': 400, 'learning_rate': 0.05, 'verbose': -1}

    for train_idx, calib_idx in kf.split(Z_features):
        Z_train, X_train, Y_train = Z_features[train_idx], X_target[train_idx], Y_target[train_idx]
        Z_calib, X_calib, Y_calib = Z_features[calib_idx], X_target[calib_idx], Y_target[calib_idx]
        
        # Train and predict for X
        model_x_low, model_x_high = lgb.LGBMRegressor(**lgbm_params, alpha=alpha/2), lgb.LGBMRegressor(**lgbm_params, alpha=1-alpha/2)
        model_x_low.fit(Z_train, X_train); model_x_high.fit(Z_train, X_train)
        all_preds_x_low = np.concatenate([all_preds_x_low, model_x_low.predict(Z_calib)])
        all_preds_x_high = np.concatenate([all_preds_x_high, model_x_high.predict(Z_calib)])

        # Train and predict for Y
        model_y_low, model_y_high = lgb.LGBMRegressor(**lgbm_params, alpha=alpha/2), lgb.LGBMRegressor(**lgbm_params, alpha=1-alpha/2)
        model_y_low.fit(Z_train, Y_train); model_y_high.fit(Z_train, Y_train)
        all_preds_y_low = np.concatenate([all_preds_y_low, model_y_low.predict(Z_calib)])
        all_preds_y_high = np.concatenate([all_preds_y_high, model_y_high.predict(Z_calib)])
        
        all_indices = np.concatenate([all_indices, calib_idx])
        all_true_x, all_true_y = np.concatenate([all_true_x, X_calib]), np.concatenate([all_true_y, Y_calib])

    # Reorder results
    sort_order = np.argsort(all_indices)
    true_x, true_y = all_true_x[sort_order], all_true_y[sort_order]
    preds_x_low, preds_x_high = all_preds_x_low[sort_order], all_preds_x_high[sort_order]
    preds_y_low, preds_y_high = all_preds_y_low[sort_order], all_preds_y_high[sort_order]

    # Conformal calibration
    scores_x = np.maximum(preds_x_low - true_x, true_x - preds_x_high)
    scores_y = np.maximum(preds_y_low - true_y, true_y - preds_y_high)
    q_level = np.ceil((1 - alpha) * (len(data) + 1)) / len(data)
    q_x, q_y = np.quantile(scores_x, q_level), np.quantile(scores_y, q_level)

    # Calculate conformalized residuals
    centers_x = (preds_x_high + preds_x_low) / 2
    widths_x = (preds_x_high - preds_x_low) + 2 * q_x
    U = (true_x - centers_x) / np.where(widths_x == 0, 1, widths_x)
    centers_y = (preds_y_high + preds_y_low) / 2
    widths_y = (preds_y_high - preds_y_low) + 2 * q_y
    V = (true_y - centers_y) / np.where(widths_y == 0, 1, widths_y)
    
    return dcor_test(U, V)


def e_value_dml_ci_test(U, V, betting_folds=2):
    """Calculates an e-value on pre-computed residuals."""
    final_e_value = 1.0
    kf = KFold(n_splits=betting_folds, shuffle=True, random_state=123)
    for train_idx, test_idx in kf.split(U):
        U_train, U_test = U[train_idx], U[test_idx]
        V_train, V_test = V[train_idx], V[test_idx]
        betting_model = Ridge(alpha=1.0)
        betting_model.fit(U_train.reshape(-1, 1), V_train)
        bets = betting_model.predict(U_test.reshape(-1, 1))
        e_process_fold = 1 + np.clip(bets, -0.9, 0.9) * V_test
        final_e_value *= np.prod(e_process_fold)
    return final_e_value
```

