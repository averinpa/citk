# Random Forest CI Test

The Random Forest Conditional Independence (CI) test is a flexible, non-parametric method that leverages the predictive power of ensemble models to assess conditional independence. By measuring the importance of a feature in a predictive task, it can capture complex, non-linear relationships and interactions, making it well-suited for a wide range of data types.

## Mathematical Formulation

The test is based on a simple but powerful premise: if a variable *X* is conditionally independent of a target *Y* given a set of conditioning variables *Z*, then *X* should have no predictive power for *Y* when the information in *Z* is already available. The test formalizes this idea by measuring the feature importance of *X* within a Random Forest model trained to predict *Y*.

The most robust measure of feature importance for this task is **permutation importance** (Breiman, 2001; Fisher et al., 2019). The overall procedure to obtain a p-value is as follows:

1.  **Train a Predictive Model**: A Random Forest model is trained to predict the target variable `Y` using the predictor `X` and the conditioning set `Z` as features.

2.  **Calculate Observed Importance**: The permutation importance of `X` is calculated. This is done by first recording the model's performance (e.g., R² for regression, accuracy for classification) on a hold-out dataset. Then, the values in the column corresponding to feature `X` are randomly shuffled (permuted), and the model's performance is measured again. The feature importance is the drop in performance caused by this shuffling. This serves as the **observed test statistic**.
    $$
    \text{Importance}(X) = \text{Performance}_{\text{original}} - \text{Performance}_{\text{permuted}(X)}
    $$

3.  **Generate a Null Distribution**: To determine if the observed importance is statistically significant, we need to generate a distribution of importances under the null hypothesis ($Y \perp X | Z$). This is achieved by permuting the relationship between *X* and *Y* while preserving the relationship with *Z*. Specifically, for a number of repetitions:
    *   The values of feature `X` are permuted.
    *   A *new* Random Forest is trained from scratch on this permuted dataset (predicting `Y` from the permuted `X` and original `Z`).
    *   The permutation importance of the (permuted) `X` is calculated for this new model.
    *   This collection of importance scores forms the **null distribution**.

4.  **Calculate P-Value**: The p-value is the proportion of importance scores in the null distribution that are greater than or equal to the originally observed importance statistic.

## Properties and Assumptions

*   **Non-parametric**: The test does not rely on assumptions of linearity or specific data distributions.
*   **Handles Interactions and Mixed Data**: Random Forests naturally handle interaction effects between variables and can be used with a mix of continuous and categorical data types.
*   **Model-Agnostic Principle**: While this implementation uses Random Forest, the underlying permutation-based testing framework is model-agnostic and can be applied with other predictive models (Fisher et al., 2019).
*   **Assumptions**: The primary assumption is that the Random Forest model is a sufficiently good predictor of the underlying relationships. If the model fails to capture the predictive patterns, the feature importance measures will not be reliable.

## Code Example

```python
import numpy as np
from citk.tests import RandomForest

# Generate data with a non-linear relationship: X -> Z -> Y
n = 500
X = np.random.randn(n)
Z = np.sin(X * 2) + np.random.randn(n) * 0.2
Y = Z**3 + np.random.randn(n) * 0.2
data = np.vstack([X, Y, Z]).T

# Initialize the test
# num_permutations can be increased for more precise p-values
rf_test = RandomForest(data, num_permutations=99, random_state=42)

# Test for unconditional independence (should be dependent)
p_unconditional = rf_test(0, 1)
print(f"P-value (unconditional) for X _||_ Y: {p_unconditional:.4f}")

# Test for conditional independence given Z (should be independent)
p_conditional = rf_test(0, 1, [2])
print(f"P-value (conditional) for X _||_ Y | Z: {p_conditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.ml_based_tests.RandomForest`.

## References

*   Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5-32.
*   Fisher, A., Rudin, C., & Dominici, F. (2019). All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models. *Journal of Machine Learning Research, 20*(177), 1-81.
*   Strobl, C., Boulesteix, A. L., Kneib, T., Augustin, T., & Zeileis, A. (2008). Conditional variable importance for random forests. *BMC Bioinformatics, 9*(1), 307. 