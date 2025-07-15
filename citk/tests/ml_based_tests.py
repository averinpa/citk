import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from typing import Optional, List
import lightgbm as lgb
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import Ridge
import dcor

from .base import CITKTest
from causallearn.utils.cit import register_ci_test, KCI as KCI_test


class KCI(CITKTest):
    """
    Wrapper for the Kernel Conditional Independence (KCI) test from the causal-learn library.
    """
    def __init__(self, data, **kwargs):
        """
        Parameters
        ----------
        data : np.ndarray
            The dataset from which to run the test.
        **kwargs : dict
            Additional keywords for the KCI test. See causal-learn documentation.
        """
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent('kci', "NO SPECIFIED PARAMETERS") # KCI handles its own params
        self.kci_instance = KCI_test(data, **kwargs)

    def __call__(self, X, Y, condition_set=None, **kwargs):
        """
        Performs the KCI test.
        """
        _, _, _, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache:
            return float(self.pvalue_cache[cache_key])
        
        p_value = self.kci_instance(X, Y, condition_set)
        
        self.pvalue_cache[cache_key] = str(p_value)
        return p_value

register_ci_test("kci", KCI)


class RandomForest(CITKTest):
    """
    Performs a conditional independence test using Random Forest feature importance.
    """
    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
        self.df = pd.DataFrame(data)
        self.df.columns = [str(i) for i in range(data.shape[1])]
        self.n_estimators = kwargs.get('n_estimators', 100)
        self.num_permutations = kwargs.get('num_permutations', 100)
        self.random_state = kwargs.get('random_state', None)
        params = f"n_est={self.n_estimators},n_perm={self.num_permutations},seed={self.random_state}"
        self.check_cache_method_consistent('rf', params)

    def __call__(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        if condition_set is None:
            condition_set = []
        else:
            # Ensure condition_set is a list, as causal-learn can pass it as a tuple
            condition_set = list(condition_set)

        _, _, _, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache:
            return float(self.pvalue_cache[cache_key])

        # Define predictor and target variables
        x_name, y_name = str(X), str(Y)
        condition_names = [str(c) for c in condition_set]
        
        predictor_cols = [x_name] + condition_names
        X_df = self.df[predictor_cols]
        y_series = self.df[y_name]
        
        # Determine if it's a classification or regression task
        is_classification = y_series.nunique() <= 10 or pd.api.types.is_categorical_dtype(y_series.dtype)
        
        if not condition_names:
            # --- Unconditional Case: Permutation test on R-squared ---
            # For the unconditional case, feature importance is always 1.0, which is not a useful metric.
            # Instead, we test the model's predictive power (R-squared) against a permuted target.
            if is_classification:
                raise NotImplementedError("Unconditional classification test with RF is not yet implemented.")
            
            model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1)
            
            # 1. Observed R-squared
            model.fit(X_df, y_series)
            observed_r2 = model.score(X_df, y_series)
            
            # 2. Null distribution of R-squared from permuted target
            permuted_r2 = np.zeros(self.num_permutations)
            y_permuted_np = y_series.to_numpy()
            for i in range(self.num_permutations):
                np.random.shuffle(y_permuted_np)
                model.fit(X_df, y_permuted_np)
                permuted_r2[i] = model.score(X_df, y_permuted_np)
                
            p_value = (np.sum(permuted_r2 >= observed_r2) + 1) / (self.num_permutations + 1)

        else:
            # --- Conditional Case: Permutation test on feature importance ---
            if is_classification:
                model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1)

        # 1. Calculate the observed statistic
        model.fit(X_df, y_series)
        importances = model.feature_importances_
        x_index = X_df.columns.get_loc(x_name)
        observed_statistic = importances[x_index]

        # 2. Generate the null distribution
        permuted_statistics = np.zeros(self.num_permutations)
        y_permuted_np = y_series.to_numpy()
        
        for i in range(self.num_permutations):
            # Permute the target variable
            np.random.shuffle(y_permuted_np)
            
            # Re-train the model on the permuted data
            model.fit(X_df, y_permuted_np)
            permuted_importances = model.feature_importances_
            permuted_statistics[i] = permuted_importances[x_index]

        # 3. Calculate the p-value
        p_value = (np.sum(permuted_statistics >= observed_statistic) + 1) / (self.num_permutations + 1)
        
        self.pvalue_cache[cache_key] = str(p_value)
        return p_value

register_ci_test("rf", RandomForest)

# Helper functions from instructions.md
def get_dml_residuals(data, x_idx, y_idx, z_idx, cv_folds=5):
    """Generates high-quality DML residuals and normalizes them."""
    X_target = data[:, x_idx]
    Y_target = data[:, y_idx]
    Z_features = data[:, z_idx] if z_idx else np.zeros((data.shape[0], 0))
    
    model = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.05, verbose=-1)
    
    if Z_features.shape[1] == 0:
        pred_x = np.zeros_like(X_target)
        pred_y = np.zeros_like(Y_target)
    else:
        pred_x = cross_val_predict(model, Z_features, X_target, cv=cv_folds)
        pred_y = cross_val_predict(model, Z_features, Y_target, cv=cv_folds)
    
    U = X_target - pred_x
    V = Y_target - pred_y
    
    # Avoid division by zero if a residual is constant
    u_std = np.std(U)
    v_std = np.std(V)
    U = U / u_std if u_std > 0 else U
    V = V / v_std if v_std > 0 else V
    
    return U, V

def dcor_test(x, y, n_perms=499):
    """
    The final-stage p-value test based on distance correlation.
    Uses the dcor library's built-in permutation test for robustness.
    """
    # Using the library's built-in permutation test is more robust
    # than a manual implementation.
    result = dcor.independence.distance_covariance_test(x, y, num_resamples=n_perms)
    return result.p_value

def conformalized_ci_test(data, x_idx, y_idx, z_idx, alpha=0.1, cv_folds=5, n_perms=199):
    """Performs the Conformalized Residual Independence Test (CRIT)."""
    X_target, Y_target = data[:, x_idx], data[:, y_idx]
    Z_features = data[:, z_idx] if z_idx else np.zeros((data.shape[0], 0))
    
    if Z_features.shape[1] == 0: # Unconditional case, just run dcor test
        return dcor_test(X_target, Y_target, n_perms=n_perms)

    all_indices, all_true_x, all_true_y = np.array([], dtype=int), np.array([]), np.array([])
    all_preds_x_low, all_preds_x_high = np.array([]), np.array([])
    all_preds_y_low, all_preds_y_high = np.array([]), np.array([])
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    lgbm_params = {'objective': 'quantile', 'metric': 'quantile', 'n_estimators': 400, 'learning_rate': 0.05, 'verbose': -1}

    for train_idx, calib_idx in kf.split(Z_features):
        Z_train, X_train, Y_train = Z_features[train_idx], X_target[train_idx], Y_target[train_idx]
        
        # Train and predict for X
        model_x_low, model_x_high = lgb.LGBMRegressor(**lgbm_params, alpha=alpha/2), lgb.LGBMRegressor(**lgbm_params, alpha=1-alpha/2)
        model_x_low.fit(Z_train, X_train); model_x_high.fit(Z_train, X_train)
        all_preds_x_low = np.concatenate([all_preds_x_low, model_x_low.predict(Z_features[calib_idx])])
        all_preds_x_high = np.concatenate([all_preds_x_high, model_x_high.predict(Z_features[calib_idx])])

        # Train and predict for Y
        model_y_low, model_y_high = lgb.LGBMRegressor(**lgbm_params, alpha=alpha/2), lgb.LGBMRegressor(**lgbm_params, alpha=1-alpha/2)
        model_y_low.fit(Z_train, Y_train); model_y_high.fit(Z_train, Y_train)
        all_preds_y_low = np.concatenate([all_preds_y_low, model_y_low.predict(Z_features[calib_idx])])
        all_preds_y_high = np.concatenate([all_preds_y_high, model_y_high.predict(Z_features[calib_idx])])
        
        all_indices = np.concatenate([all_indices, calib_idx])
        all_true_x, all_true_y = np.concatenate([all_true_x, X_target[calib_idx]]), np.concatenate([all_true_y, Y_target[calib_idx]])

    sort_order = np.argsort(all_indices)
    true_x, true_y = all_true_x[sort_order], all_true_y[sort_order]
    preds_x_low, preds_x_high = all_preds_x_low[sort_order], all_preds_x_high[sort_order]
    preds_y_low, preds_y_high = all_preds_y_low[sort_order], all_preds_y_high[sort_order]

    scores_x = np.maximum(preds_x_low - true_x, true_x - preds_x_high)
    scores_y = np.maximum(preds_y_low - true_y, true_y - preds_y_high)
    q_level = np.ceil((1 - alpha) * (len(data) + 1)) / len(data)
    q_x, q_y = np.quantile(scores_x, q_level), np.quantile(scores_y, q_level)

    centers_x = (preds_x_high + preds_x_low) / 2
    widths_x = (preds_x_high - preds_x_low) + 2 * q_x
    U = (true_x - centers_x) / np.where(widths_x == 0, 1, widths_x)
    centers_y = (preds_y_high + preds_y_low) / 2
    widths_y = (preds_y_high - preds_y_low) + 2 * q_y
    V = (true_y - centers_y) / np.where(widths_y == 0, 1, widths_y)
    
    return dcor_test(U, V, n_perms=n_perms)

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

class DML(CITKTest):
    """Double-ML based conditional independence test."""
    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
        self.cv_folds = kwargs.get('cv_folds', 5)
        self.n_perms = kwargs.get('n_perms', 199)
        self.check_cache_method_consistent('dml', f"cv={self.cv_folds},n_perms={self.n_perms}")

    def __call__(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        if condition_set is None: condition_set = []
        else: condition_set = list(condition_set)
        
        _, _, _, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache:
            return float(self.pvalue_cache[cache_key])

        U, V = get_dml_residuals(self.data, X, Y, condition_set, cv_folds=self.cv_folds)
        p_value = dcor_test(U, V, n_perms=self.n_perms)

        self.pvalue_cache[cache_key] = str(p_value)
        return p_value

register_ci_test("dml", DML)

class CRIT(CITKTest):
    """Conformalized Residual Independence Test (CRIT)."""
    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
        self.alpha = kwargs.get('alpha', 0.1)
        self.cv_folds = kwargs.get('cv_folds', 5)
        self.n_perms = kwargs.get('n_perms', 199)
        params = f"alpha={self.alpha},cv={self.cv_folds},n_perms={self.n_perms}"
        self.check_cache_method_consistent('crit', params)

    def __call__(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        if condition_set is None: condition_set = []
        else: condition_set = list(condition_set)
        
        _, _, _, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache:
            return float(self.pvalue_cache[cache_key])

        p_value = conformalized_ci_test(self.data, X, Y, condition_set, 
                                        alpha=self.alpha, cv_folds=self.cv_folds, n_perms=self.n_perms)

        self.pvalue_cache[cache_key] = str(p_value)
        return p_value

register_ci_test("crit", CRIT)

class EDML(CITKTest):
    """E-Value Double-ML based conditional independence test."""
    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
        self.cv_folds = kwargs.get('cv_folds', 5)
        self.betting_folds = kwargs.get('betting_folds', 2)
        params = f"cv={self.cv_folds},bet_folds={self.betting_folds}"
        self.check_cache_method_consistent('edml', params)

    def __call__(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        if condition_set is None: condition_set = []
        else: condition_set = list(condition_set)
        
        _, _, _, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache:
            return float(self.pvalue_cache[cache_key])

        U, V = get_dml_residuals(self.data, X, Y, condition_set, cv_folds=self.cv_folds)
        e_value = e_value_dml_ci_test(U, V, betting_folds=self.betting_folds)
        
        # Convert e-value to p-value. 1/e is a common (though sometimes conservative) choice.
        # Ensure p-value is at most 1.
        p_value = min(1.0, 1.0 / e_value if e_value > 0 else float('inf'))

        self.pvalue_cache[cache_key] = str(p_value)
        return p_value

register_ci_test("edml", EDML) 