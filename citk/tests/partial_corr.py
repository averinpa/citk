import numpy as np
from scipy.stats import pearsonr
from typing import Optional, cast
from causallearn.utils.cit import register_ci_test, NO_SPECIFIED_PARAMETERS_MSG
from .base import CITKTest

class PartialCorrelation(CITKTest):
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
        # This call is necessary to set self.method and integrate with the framework
        self.check_cache_method_consistent('partial_corr', NO_SPECIFIED_PARAMETERS_MSG)

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

        data = self.data
        if not condition_set:
            # The pearsonr function returns a tuple (statistic, pvalue)
            p_value = cast(float, pearsonr(data[:, X], data[:, Y])[1])
        else:
            # Prepare data for regression
            z_data = data[:, condition_set]
            x_data = data[:, X]
            y_data = data[:, Y]

            # Add intercept term
            z_data_with_intercept = np.hstack([np.ones((z_data.shape[0], 1)), z_data])
            
            # Regress X on Z
            beta_x = np.linalg.lstsq(z_data_with_intercept, x_data, rcond=None)[0]
            res_x = x_data - z_data_with_intercept @ beta_x

            # Regress Y on Z
            beta_y = np.linalg.lstsq(z_data_with_intercept, y_data, rcond=None)[0]
            res_y = y_data - z_data_with_intercept @ beta_y
            
            # Compute Pearson correlation on residuals
            p_value = cast(float, pearsonr(res_x, res_y)[1])
        
        self.pvalue_cache[cache_key] = str(p_value)
        return p_value

# Register the custom test with causal-learn so it can be called by name
register_ci_test("partial_corr", PartialCorrelation)
