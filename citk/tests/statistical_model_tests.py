import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from typing import Optional, List

from .base import CITKTest
from causallearn.utils.cit import register_ci_test, NO_SPECIFIED_PARAMETERS_MSG

def _glm_conditional_independence_test(df: pd.DataFrame, x: int, y: int, z: List[int],
                                       model_class, family=None, **kwargs):
    # Convert integer indices to string column names for the formula
    x_name = f'v{x}'
    y_name = f'v{y}'
    z_names = [f'v{i}' for i in z]
    df.columns = [f'v{i}' for i in range(df.shape[1])]

    # --- Null Model (H0: y ~ z) ---
    if z:
        formula_null = f"{y_name} ~ {' + '.join(z_names)}"
    else: # If conditioning set is empty, null model is intercept-only
        formula_null = f"{y_name} ~ 1"

    model_args = {'formula': formula_null, 'data': df}
    if family:
        model_args['family'] = family

    model_null = model_class.from_formula(**model_args, **kwargs).fit()
    loglik_null = model_null.llf
    df_null = model_null.df_model # Number of predictors in the null model

    # --- Alternative Model (H1: y ~ x + z) ---
    all_predictors = [x_name] + z_names
    formula_alt = f"{y_name} ~ {' + '.join(all_predictors)}"

    model_args['formula'] = formula_alt
    model_alt = model_class.from_formula(**model_args, **kwargs).fit()
    loglik_alt = model_alt.llf
    df_alt = model_alt.df_model # Number of predictors in the alternative model

    # --- Likelihood Ratio Test ---
    lr_stat = 2 * (loglik_alt - loglik_null)
    df_diff = df_alt - df_null

    if lr_stat < 0: lr_stat = 0
    if df_diff <= 0: return 1.0

    p_value = chi2.sf(lr_stat, df=df_diff)
    return p_value

class Regression(CITKTest):
    """CI test for continuous targets using Linear Regression (OLS)."""
    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
        self.df = pd.DataFrame(data)
        self.check_cache_method_consistent('reg', NO_SPECIFIED_PARAMETERS_MSG)

    def __call__(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        if condition_set is None: condition_set = []
        _, _, _, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache:
            return float(self.pvalue_cache[cache_key])

        p_value = _glm_conditional_independence_test(self.df, X, Y, condition_set, model_class=sm.OLS)
        self.pvalue_cache[cache_key] = str(p_value)
        return float(p_value)

register_ci_test("reg", Regression)

class Logit(CITKTest):
    """CI test for binary targets using Logistic Regression."""
    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
        self.df = pd.DataFrame(data)
        self.check_cache_method_consistent('logit', NO_SPECIFIED_PARAMETERS_MSG)

    def __call__(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        if condition_set is None: condition_set = []
        _, _, _, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache:
            return float(self.pvalue_cache[cache_key])

        p_value = _glm_conditional_independence_test(self.df, X, Y, condition_set, model_class=sm.Logit)
        self.pvalue_cache[cache_key] = str(p_value)
        return float(p_value)

register_ci_test("logit", Logit)

class Poisson(CITKTest):
    """CI test for count targets using Poisson Regression."""
    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
        self.df = pd.DataFrame(data)
        self.check_cache_method_consistent('pois', NO_SPECIFIED_PARAMETERS_MSG)

    def __call__(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        if condition_set is None: condition_set = []
        _, _, _, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache:
            return float(self.pvalue_cache[cache_key])

        p_value = _glm_conditional_independence_test(self.df, X, Y, condition_set, model_class=sm.GLM, family=sm.families.Poisson())
        self.pvalue_cache[cache_key] = str(p_value)
        return float(p_value)

register_ci_test("pois", Poisson) 