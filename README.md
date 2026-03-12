# hsic-lasso-selector

[![PyPI version](https://badge.fury.io/py/hsic-lasso-selector.svg)](https://pypi.org/project/hsic-lasso-selector/)
[![Python versions](https://img.shields.io/pypi/pyversions/hsic-lasso-selector.svg)](https://pypi.org/project/hsic-lasso-selector/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **scikit-learn compatible** feature selector based on
[HSIC Lasso](https://github.com/riken-aip/pyHSICLasso) —
a kernel-based method that selects features maximally dependent on the
target while being minimally redundant with each other.

---

## Quick start

```python
import numpy as np
from hsic_lasso_selector import HSICLassoSelector

rng = np.random.default_rng(0)
X = rng.standard_normal((300, 50))
y = 3 * X[:, 0] - 2 * X[:, 7] + rng.standard_normal(300) * 0.1

sel = HSICLassoSelector(num_feat=5, B=50, random_state=42)
sel.fit(X, y)

print(sel.selected_indices_)           # e.g. [0, 7, ...]
print(sel.feature_importances_)        # length-50 array
sel.summary()                          # ranked table

X_selected = sel.transform(X)         # shape (300, 5)
```

### Drop-in replacement for the original pyHSICLasso usage

```python
# Before
from pyHSICLasso import HSICLasso
hsic_lasso = HSICLasso()
hsic_lasso.input(X, y.ravel(), featname=feature_names)
hsic_lasso.regression(num_feat=6, B=50)
selected_names  = feature_names[hsic_lasso.get_index()]
hsic_coef       = hsic_lasso.beta.ravel()
importances     = np.zeros(hsic_coef.shape)
importances[hsic_lasso.get_index()] = hsic_lasso.get_index_score()

# After
from hsic_lasso_selector import HSICLassoSelector
sel = HSICLassoSelector(num_feat=6, B=50, feature_names=feature_names)
sel.fit(X, y)
selected_names = sel.get_feature_names_out()
importances    = sel.feature_importances_
```

---

## Pipeline & GridSearchCV

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ("sel", HSICLassoSelector(task="regression")),
    ("reg", Ridge()),
])

grid = GridSearchCV(
    pipe,
    param_grid={
        "sel__num_feat": [5, 10, 20],
        "sel__B":        [0, 50],
    },
    cv=5,
    scoring="r2",
)
grid.fit(X_train, y_train)
print(grid.best_params_)
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_feat` | int | `5` | Number of features to select |
| `B` | int | `20` | Block size (`0` = exact vanilla HSIC; `20`–`100` for scalable block mode) |
| `M` | int | `3` | Block permutations for stability (when `B > 0`) |
| `task` | str | `"regression"` | `"regression"` or `"classification"` |
| `n_jobs` | int | `-1` | Parallel threads (`-1` = all cores) |
| `discrete_x` | bool | `False` | Use Delta kernel for all features (categorical inputs) |
| `max_neighbors` | int | `10` | Max neighbours for the related-feature summary graph |
| `covars` | ndarray\|None | `None` | Optional covariate matrix to condition on |
| `covars_kernel` | str | `"Gaussian"` | Kernel for the covariate matrix (`"Gaussian"` or `"Delta"`) |
| `feature_names` | list\|None | `None` | Human-readable feature names |
| `random_state` | int\|None | `None` | Seed for reproducibility |

## Fitted attributes

| Attribute | Shape | Description |
|---|---|---|
| `selected_indices_` | `(num_feat,)` | Column indices of selected features |
| `feature_importances_` | `(n_features_in_,)` | Full-length importance vector |
| `hsic_scores_` | `(num_feat,)` | HSIC scores for selected features |
| `beta_` | `(n_features_in_,)` | Raw pyHSICLasso beta vector |
| `hsic_lasso_` | `HSICLasso` | Raw pyHSICLasso object for advanced use |

---

## Pandas / `set_output` support

```python
import pandas as pd

df = pd.DataFrame(X, columns=[f"gene_{i}" for i in range(50)])
sel = HSICLassoSelector(num_feat=5, B=50, random_state=0)
sel.set_output(transform="pandas")
sel.fit(df, y)
df_selected = sel.transform(df)   # returns a DataFrame with correct column names
```

---

## Development

```bash
git clone https://github.com/yourname/hsic-lasso-selector
cd hsic-lasso-selector
pip install -e ".[dev]"
pytest tests/ -v --cov=hsic_lasso_selector
```

---

## References

Yamada, M., Jitkrittum, W., Sigal, L., Xing, E. P., & Sugiyama, M. (2014).
High-dimensional feature selection by feature-wise kernelized lasso.
*Neural Computation*, 26(1), 185–207.

---

## License

MIT — see [LICENSE](LICENSE).
