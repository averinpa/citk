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

    model_null = model_class.from_formula(**model_args, **kwargs).fit(disp=0)
    loglik_null = model_null.llf
    df_null = model_null.df_model # Number of predictors in the null model

    # --- Alternative Model (H1: y ~ x + z) ---
    all_predictors = [x_name] + z_names
    formula_alt = f"{y_name} ~ {' + '.join(all_predictors)}"

    model_args['formula'] = formula_alt
    model_alt = model_class.from_formula(**model_args, **kwargs).fit(disp=0)
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
        """
        Performs a conditional independence test for continuous data using a likelihood-ratio test
        between nested Ordinary Least Squares (OLS) linear regression models.

        Parameters
        ----------
        X : int
            The index of the first variable.
        Y : int
            The index of the second variable (the target).
        condition_set : list[int], optional
            A list of indices for the conditioning set. Can be empty.

        Returns
        -------
        p_value : float
            The p-value of the test.


    .. seealso::
        For a detailed explanation of the statistical test, including mathematical
        formulations and assumptions, please refer to the :doc:`/tests/regression_test` guide.

    Examples
    --------
    **Standalone Usage**

    .. code-block:: python

        import numpy as np
        from citk.tests import Regression

        # Generate data where X and Y are independent given Z
        # X -> Z -> Y
        n = 500
        X = np.random.randn(n)
        Z = 2 * X + np.random.randn(n)
        Y = 3 * Z + np.random.randn(n)
        data = np.vstack([X, Y, Z]).T

        # Initialize the test
        regression_test = Regression(data)

        # Test if X and Y are independent
        p_value_unconditional = regression_test(0, 1)
        print(f"P-value (unconditional) for X _||_ Y: {p_value_unconditional:.4f}")

        # Test if X and Y are independent given Z
        p_value_conditional = regression_test(0, 1, [2])
        print(f"P-value (conditional) for X _||_ Y | Z: {p_value_conditional:.4f}")

    .. code-block:: text

        P-value (unconditional) for X _||_ Y: 0.0000
        P-value (conditional) for X _||_ Y | Z: 0.6210

    **Usage with PC Algorithm**

    .. code-block:: python

        from causallearn.search.ConstraintBased.PC import pc
        from citk.tests import Regression # Make sure it's registered

        # Re-use the same data from the standalone example
        cg = pc(data, alpha=0.05, indep_test='reg')

        print("Estimated Causal Graph:")
        print(cg.G)

    .. code-block:: text

        Estimated Causal Graph:
        Graph Nodes:
        X1;X2;X3

        Graph Edges:
        1. X1 --- X3
        2. X2 --- X3
        """
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
        """
        Performs a conditional independence test for binary data using a likelihood-ratio test
        between nested Logistic Regression models.

        Parameters
        ----------
        X : int
            The index of the first variable.
        Y : int
            The index of the second variable (the binary target).
        condition_set : list[int], optional
            A list of indices for the conditioning set. Can be empty.

        Returns
        -------
        p_value : float
            The p-value of the test.


    .. seealso::
        For a detailed explanation of the statistical test, including mathematical
        formulations and assumptions, please refer to the :doc:`/tests/logit_test` guide.

    Examples
    --------
    **Standalone Usage**

    .. code-block:: python

        import numpy as np
        from citk.tests import Logit

        # Generate data where Y is a binary variable
        # X -> Z -> Y
        n = 500
        X = np.random.randn(n)
        Z = 2 * X + np.random.randn(n)
        Y = (3 * Z + np.random.randn(n)) > 0
        data = np.vstack([X, Y, Z]).T

        # Initialize the test
        logit_test = Logit(data)

        # Test if X and Y are independent
        p_value_unconditional = logit_test(0, 1)
        print(f"P-value (unconditional) for X _||_ Y: {p_value_unconditional:.4f}")

        # Test for conditional independence of X and Y given Z
        p_value_conditional = logit_test(0, 1, [2])
        print(f"P-value (conditional) for X _||_ Y | Z: {p_value_conditional:.4f}")

    .. code-block:: text

        P-value (unconditional) for X _||_ Y: 0.0000
        P-value (conditional) for X _||_ Y | Z: 0.9388

    **Usage with PC Algorithm**

    .. code-block:: python

        from causallearn.search.ConstraintBased.PC import pc
        from citk.tests import Logit
        import numpy as np

        # For the PC algorithm example, we use fully binary data to ensure
        # the Logit test is always applicable, as PC may test any variable pair.
        # We model a causal chain X -> Z -> Y with some noise.
        n = 500
        X = np.random.randint(0, 2, n)
        # Z depends on X, with a 10% chance of flipping
        Z = X.copy()
        flip_mask_z = np.random.random(n) < 0.1
        Z[flip_mask_z] = 1 - Z[flip_mask_z]
        # Y depends on Z, with a 10% chance of flipping
        Y = Z.copy()
        flip_mask_y = np.random.random(n) < 0.1
        Y[flip_mask_y] = 1 - Y[flip_mask_y]
        binary_data = np.vstack([X, Y, Z]).T

        cg = pc(binary_data, alpha=0.05, indep_test='logit')

        print("Estimated Causal Graph:")
        print(cg.G)

    .. code-block:: text

        Estimated Causal Graph:
        Graph Nodes:
        X1;X2;X3

        Graph Edges:
        1. X1 --- X3
        2. X2 --- X3
        """
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
        """
        Performs a conditional independence test for count data using a likelihood-ratio test
        between nested Poisson Regression models.

        Parameters
        ----------
        X : int
            The index of the first variable.
        Y : int
            The index of the second variable (the count target).
        condition_set : list[int], optional
            A list of indices for the conditioning set. Can be empty.

        Returns
        -------
        p_value : float
            The p-value of the test.
        

    .. seealso::
        For a detailed explanation of the statistical test, including mathematical
        formulations and assumptions, please refer to the :doc:`/tests/poisson_test` guide.

    Examples
    --------
    **Standalone Usage**

    .. code-block:: python

        import numpy as np
        from citk.tests import Poisson

        # Generate data where Y is a count variable
        # X -> Z -> Y
        n = 500
        X = np.random.randn(n)
        Z = 0.5 * X + np.random.randn(n)
        Y = np.random.poisson(np.exp(1 + 0.5 * Z))
        data = np.vstack([X, Y, Z]).T

        # Initialize the test
        poisson_test = Poisson(data)

        # Test if X and Y are independent
        p_value_unconditional = poisson_test(0, 1)
        print(f"P-value (unconditional) for X _||_ Y: {p_value_unconditional:.4f}")

        # Test for conditional independence of X and Y given Z
        p_value_conditional = poisson_test(0, 1, [2])
        print(f"P-value (conditional) for X _||_ Y | Z: {p_value_conditional:.4f}")

    .. code-block:: text

        P-value (unconditional) for X _||_ Y: 0.0000
        P-value (conditional) for X _||_ Y | Z: 0.2017

    **Usage with PC Algorithm**

    .. code-block:: python

        from causallearn.search.ConstraintBased.PC import pc
        from citk.tests import Poisson
        import numpy as np

        # For the PC algorithm example, we use fully count-based data to
        # ensure the Poisson test is always applicable.
        n = 500
        X = np.random.poisson(2, size=n)
        Z = np.random.poisson(1 + X / 2)
        Y = np.random.poisson(1 + Z / 2)
        count_data = np.vstack([X, Y, Z]).T

        cg = pc(count_data, alpha=0.05, indep_test='pois')

        print("Estimated Causal Graph:")
        print(cg.G)

    .. code-block:: text

        Estimated Causal Graph:
        Graph Nodes:
        X1;X2;X3

        Graph Edges:
        1. X1 --- X3
        2. X2 --- X3
        """
        if condition_set is None: condition_set = []
        _, _, _, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache:
            return float(self.pvalue_cache[cache_key])

        p_value = _glm_conditional_independence_test(self.df, X, Y, condition_set, model_class=sm.GLM, family=sm.families.Poisson())
        self.pvalue_cache[cache_key] = str(p_value)
        return float(p_value)

register_ci_test("pois", Poisson) 