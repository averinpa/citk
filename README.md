# `citk`: A Conditional Independence Test Toolkit

`citk` is a Python library that provides a comprehensive and modern toolkit for conditional independence (CI) testing. It is designed to be seamlessly integrated with the [`causal-learn`](https://github.com/cmu-phil/causal-learn) package and offers a collection of classical, statistical, and advanced machine learning-based CI tests.

The library is structured to be a powerful benchmark for causal discovery and a practical toolkit for researchers and practitioners.

## Features

- **Wide Range of Tests**: Includes classical tests (Fisher's Z, G-Squared), statistical model-based tests (GLM-based), and modern ML-based tests (KCI, Random Forest, DoubleML, and more).
- **`causal-learn` Compatible**: All tests are designed as drop-in replacements for the standard tests in the `causal-learn` ecosystem, allowing you to easily use them with algorithms like PC.

## Installation

You can install `citk` directly from this GitHub repository using `pip`:

```bash
pip install git+https://github.com/averinpa/citk.git
```

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
# You can use "fisherz", "spearman", "gsq", "rf", "dml", etc.
cg = pc(data, alpha=0.05, indep_test='spearman')

# 3. View the learned graph
print("Learned Graph Edges:")
print(cg.G.get_edges())
```

## Available Tests

- **Simple Tests**: `fisherz`, `spearman`, `gsq`, `chisq`, `dcor`
- **Statistical Model Tests**: `reg`, `logit`, `pois`
- **ML-Based Tests**: `kci`, `rf`, `dml`, `crit`, `edml`

For detailed documentation on each test and its parameters, please see our full documentation page [HERE](https://averinpa.github.io/citk/).