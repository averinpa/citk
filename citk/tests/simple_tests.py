import numpy as np
import pandas as pd
import pingouin as pg
from typing import Optional, cast

from causallearn.utils.cit import register_ci_test, NO_SPECIFIED_PARAMETERS_MSG, Chisq_or_Gsq
from .base import CITKTest


class FisherZ(CITKTest):
    """
    A custom conditional independence test for partial correlation.
    This is functionally equivalent to Fisher-Z for continuous data.
    """
    def __init__(self, data: np.ndarray, **kwargs):
        """
        Parameters
        ----------
        data : np.ndarray
            The dataset from which to run the test.
        """
        super().__init__(data, **kwargs)
        self.df = pd.DataFrame(data)
        self.df.columns = [str(i) for i in range(data.shape[1])]
        self.check_cache_method_consistent('fisherz', NO_SPECIFIED_PARAMETERS_MSG)

    def __call__(self, X: int, Y: int, condition_set: Optional[list[int]] = None, **kwargs) -> float:
        """
        Performs a partial correlation test.

        Parameters
        ----------
        X : int
            The index of the first variable.
        Y : int
            The index of the second variable.
        condition_set : list[int], optional
            A list of indices for the conditioning set. Can be empty.

        Returns
        -------
        p_value : float
            The p-value of the test.
        """
        if condition_set is None:
            condition_set = []

        _, _, _, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache:
            return float(self.pvalue_cache[cache_key])

        x_name, y_name = str(X), str(Y)
        condition_names = [str(c) for c in condition_set] if condition_set else None

        if not condition_names:
            result = pg.corr(self.df[x_name], self.df[y_name], method='pearson')
            p_value = cast(pd.Series, result['p-val']).to_numpy()[0]
        else:
            result = pg.partial_corr(data=self.df, x=x_name, y=y_name, covar=condition_names, method='pearson')
            p_value = cast(pd.Series, result['p-val']).to_numpy()[0]

        self.pvalue_cache[cache_key] = str(p_value)
        return p_value

register_ci_test("fisherz", FisherZ)


class Spearman(CITKTest):
    """
    Wrapper for the Spearman partial correlation test from the pingouin library.
    """
    def __init__(self, data: np.ndarray, **kwargs):
        """
        Parameters
        ----------
        data : np.ndarray
            The dataset from which to run the test.
        """
        super().__init__(data, **kwargs)
        self.df = pd.DataFrame(data)
        self.df.columns = [str(i) for i in range(data.shape[1])]
        self.check_cache_method_consistent('spearman', NO_SPECIFIED_PARAMETERS_MSG)

    def __call__(self, X: int, Y: int, condition_set: Optional[list[int]] = None, **kwargs) -> float:
        """
        Performs a Spearman partial correlation test.

        Parameters
        ----------
        X : int
            The index of the first variable.
        Y : int
            The index of the second variable.
        condition_set : list[int], optional
            A list of indices for the conditioning set. Can be empty.

        Returns
        -------
        p_value : float
            The p-value of the test.
        """
        if condition_set is None:
            condition_set = []

        _, _, _, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache:
            return float(self.pvalue_cache[cache_key])
        
        x_name, y_name = str(X), str(Y)
        condition_names = [str(c) for c in condition_set] if condition_set else None
        
        if not condition_names:
            result = pg.corr(self.df[x_name], self.df[y_name], method='spearman')
            p_value = cast(pd.Series, result['p-val']).to_numpy()[0]
        else:
            result = pg.partial_corr(data=self.df, x=x_name, y=y_name, covar=condition_names, method='spearman')
            p_value = cast(pd.Series, result['p-val']).to_numpy()[0]

        self.pvalue_cache[cache_key] = str(p_value)
        return p_value

register_ci_test("spearman", Spearman)


class GSq(CITKTest):
    """
    Wrapper for the G-Square test from the causal-learn library.
    This test is suitable for discrete data.
    """
    def __init__(self, data, **kwargs):
        """
        Parameters
        ----------
        data : np.ndarray
            The dataset from which to run the test.
        **kwargs : dict
            Additional keywords for the test.
        """
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent('gsq', "NO SPECIFIED PARAMETERS")
        self.test_instance = Chisq_or_Gsq(data, method_name='gsq', **kwargs)

    def __call__(self, X, Y, condition_set=None, **kwargs):
        """
        Performs the G-Square test.
        """
        _, _, _, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache:
            return float(self.pvalue_cache[cache_key])

        p_value = self.test_instance(X, Y, condition_set)

        self.pvalue_cache[cache_key] = str(p_value)
        return p_value

register_ci_test("gsq", GSq)


class ChiSq(CITKTest):
    """
    Wrapper for the Chi-Square test from the causal-learn library.
    This test is suitable for discrete data.
    """
    def __init__(self, data, **kwargs):
        """
        Parameters
        ----------
        data : np.ndarray
            The dataset from which to run the test.
        **kwargs : dict
            Additional keywords for the test.
        """
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent('chisq', "NO SPECIFIED PARAMETERS")
        self.test_instance = Chisq_or_Gsq(data, method_name='chisq', **kwargs)

    def __call__(self, X, Y, condition_set=None, **kwargs):
        """
        Performs the Chi-Square test.
        """
        _, _, _, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache:
            return float(self.pvalue_cache[cache_key])

        p_value = self.test_instance(X, Y, condition_set)

        self.pvalue_cache[cache_key] = str(p_value)
        return p_value

register_ci_test("chisq", ChiSq)


class DCor(CITKTest):
    """
    Wrapper for the distance correlation test from the pingouin library.
    NOTE: This implementation only supports unconditional tests.
    """
    def __init__(self, data: np.ndarray, **kwargs):
        """
        Parameters
        ----------
        data : np.ndarray
            The dataset from which to run the test.
        n_boot : int
            Number of bootstraps for permutation test.
        """
        super().__init__(data, **kwargs)
        self.n_boot = kwargs.get('n_boot', 1000)
        self.seed = kwargs.get('seed', None)
        self.check_cache_method_consistent('dcor', f"n_boot={self.n_boot}, seed={self.seed}")

    def __call__(self, X: int, Y: int, condition_set: Optional[list[int]] = None, **kwargs) -> float:
        """
        Performs a distance correlation test.
        """
        if condition_set:
            raise NotImplementedError("Conditional distance correlation test is not supported.")

        _, _, _, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache:
            return float(self.pvalue_cache[cache_key])

        x_data = self.data[:, X]
        y_data = self.data[:, Y]

        dcor, p_value = pg.distance_corr(x_data, y_data, n_boot=self.n_boot, seed=self.seed)

        self.pvalue_cache[cache_key] = str(p_value)
        return p_value

register_ci_test("dcor", DCor) 