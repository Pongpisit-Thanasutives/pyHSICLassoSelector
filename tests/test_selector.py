"""
Tests for HSICLassoSelector.

Run with:
    pytest tests/ -v
or:
    pytest tests/ -v --cov=hsic_lasso_selector
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_regression(n=200, p=20, n_informative=3, noise=0.05, seed=0):
    """Generate a synthetic regression dataset with known informative features."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    coef = np.zeros(p)
    coef[:n_informative] = rng.uniform(1.5, 3.0, size=n_informative)
    y = X @ coef + rng.standard_normal(n) * noise
    return X, y, np.where(coef != 0)[0]


def _make_classification(n=200, p=20, n_informative=3, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    coef = np.zeros(p)
    coef[:n_informative] = rng.uniform(1.5, 3.0, size=n_informative)
    logits = X @ coef
    y = (logits > np.median(logits)).astype(int)
    return X, y


# Shorthand defaults used across tests
_FIT = dict(B=20, M=3, random_state=0)


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

pytest.importorskip("pyHSICLasso", reason="pyHSICLasso not installed")

from hsic_lasso_selector import HSICLassoSelector  # noqa: E402


# ---------------------------------------------------------------------------
# Basic fit / transform
# ---------------------------------------------------------------------------

class TestFitTransform:
    def test_fit_returns_self(self):
        X, y, _ = _make_regression()
        sel = HSICLassoSelector(num_feat=3, **_FIT)
        result = sel.fit(X, y)
        assert result is sel

    def test_transform_shape(self):
        X, y, _ = _make_regression()
        sel = HSICLassoSelector(num_feat=3, **_FIT)
        X_t = sel.fit_transform(X, y)
        assert X_t.shape == (X.shape[0], 3)

    def test_support_indices_length(self):
        X, y, _ = _make_regression()
        sel = HSICLassoSelector(num_feat=4, **_FIT)
        sel.fit(X, y)
        assert len(sel.support_indices_) == 4

    def test_feature_importances_shape(self):
        X, y, _ = _make_regression()
        sel = HSICLassoSelector(num_feat=4, **_FIT)
        sel.fit(X, y)
        assert sel.feature_importances_.shape == (X.shape[1],)

    def test_feature_importances_nonnegative(self):
        X, y, _ = _make_regression()
        sel = HSICLassoSelector(num_feat=4, **_FIT)
        sel.fit(X, y)
        assert np.all(sel.feature_importances_ >= 0)

    def test_importances_nonzero_only_at_selected(self):
        X, y, _ = _make_regression()
        sel = HSICLassoSelector(num_feat=4, **_FIT)
        sel.fit(X, y)
        mask = sel.get_support()
        assert np.all(sel.feature_importances_[~mask] == 0.0)
        assert np.all(sel.feature_importances_[mask] > 0.0)


# ---------------------------------------------------------------------------
# get_support / get_feature_names_out
# ---------------------------------------------------------------------------

class TestSupportAndNames:
    def test_get_support_bool(self):
        X, y, _ = _make_regression()
        sel = HSICLassoSelector(num_feat=3, **_FIT).fit(X, y)
        mask = sel.get_support()
        assert mask.dtype == bool
        assert mask.shape == (X.shape[1],)
        assert mask.sum() == 3

    def test_get_support_indices(self):
        X, y, _ = _make_regression()
        sel = HSICLassoSelector(num_feat=3, **_FIT).fit(X, y)
        idx = sel.get_support(indices=True)
        assert idx.dtype in (np.int32, np.int64)
        assert len(idx) == 3

    def test_get_feature_names_out_auto(self):
        X, y, _ = _make_regression(p=10)
        sel = HSICLassoSelector(num_feat=2, **_FIT).fit(X, y)
        names = sel.get_feature_names_out()
        assert len(names) == 2
        for n in names:
            assert n.startswith("x")

    def test_get_feature_names_out_custom(self):
        X, y, _ = _make_regression(p=5)
        feat_names = [f"gene_{i}" for i in range(5)]
        sel = HSICLassoSelector(num_feat=2, **_FIT, feature_names=feat_names).fit(X, y)
        names = sel.get_feature_names_out()
        assert all(n.startswith("gene_") for n in names)

    def test_get_feature_names_out_input_features(self):
        X, y, _ = _make_regression(p=5)
        sel = HSICLassoSelector(num_feat=2, **_FIT).fit(X, y)
        custom = [f"feat_{i}" for i in range(5)]
        names = sel.get_feature_names_out(input_features=custom)
        assert all(n.startswith("feat_") for n in names)


class TestClassification:
    def test_classification_fits(self):
        X, y = _make_classification()
        sel = HSICLassoSelector(num_feat=3, task="classification", **_FIT)
        sel.fit(X, y)
        assert len(sel.support_indices_) == 3


class TestEdgeCases:
    def test_num_feat_clamped_to_n_features(self):
        X, y, _ = _make_regression(p=5)
        sel = HSICLassoSelector(num_feat=100, B=0, random_state=0)
        with pytest.warns(UserWarning, match="clamped"):
            sel.fit(X, y)
        assert len(sel.support_indices_) <= 5

    def test_wrong_n_features_transform(self):
        X, y, _ = _make_regression(p=10)
        sel = HSICLassoSelector(num_feat=3, **_FIT).fit(X, y)
        X_bad = np.random.standard_normal((50, 7))
        with pytest.raises(ValueError, match="features"):
            sel.transform(X_bad)

    def test_bad_task_raises(self):
        X, y, _ = _make_regression()
        sel = HSICLassoSelector(num_feat=3, task="unsupported")
        with pytest.raises(ValueError, match="task"):
            sel.fit(X, y)

    def test_bad_feature_names_length(self):
        X, y, _ = _make_regression(p=10)
        sel = HSICLassoSelector(num_feat=3, **_FIT, feature_names=["a", "b"])
        with pytest.raises(ValueError, match="feature_names"):
            sel.fit(X, y)

    def test_not_fitted_raises(self):
        from sklearn.exceptions import NotFittedError
        sel = HSICLassoSelector(num_feat=3)
        with pytest.raises(NotFittedError):
            sel.transform(np.zeros((10, 5)))


class TestReproducibility:
    def test_same_seed_same_result(self):
        X, y, _ = _make_regression(seed=99)
        sel1 = HSICLassoSelector(num_feat=3, **_FIT).fit(X, y)
        sel2 = HSICLassoSelector(num_feat=3, **_FIT).fit(X, y)
        np.testing.assert_array_equal(sel1.support_indices_, sel2.support_indices_)


class TestSklearnIntegration:
    def test_pipeline(self):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline

        X, y, _ = _make_regression()
        pipe = Pipeline([
            ("sel", HSICLassoSelector(num_feat=4, **_FIT)),
            ("reg", Ridge()),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_get_params_set_params(self):
        sel = HSICLassoSelector(num_feat=5, B=20)
        params = sel.get_params()
        assert params["num_feat"] == 5
        sel.set_params(num_feat=8)
        assert sel.num_feat == 8

    def test_clone(self):
        from sklearn.base import clone
        sel = HSICLassoSelector(num_feat=5, B=20, task="regression")
        clone(sel)

    def test_pandas_dataframe_input(self):
        pd = pytest.importorskip("pandas")
        X, y, _ = _make_regression(p=8)
        cols = [f"feat_{i}" for i in range(8)]
        df = pd.DataFrame(X, columns=cols)
        sel = HSICLassoSelector(num_feat=3, **_FIT)
        sel.fit(df, y)
        assert sel.n_features_in_ == 8
        X_t = sel.transform(df)
        assert X_t.shape[1] == 3

    def test_set_output_pandas(self):
        pd = pytest.importorskip("pandas")
        X, y, _ = _make_regression(p=8)
        cols = [f"feat_{i}" for i in range(8)]
        df = pd.DataFrame(X, columns=cols)
        sel = HSICLassoSelector(num_feat=3, **_FIT)
        sel.set_output(transform="pandas")
        sel.fit(df, y)
        out = sel.transform(df)
        assert hasattr(out, "columns"), "Expected a DataFrame output"
        assert out.shape[1] == 3


class TestSummary:
    def test_summary_returns_string(self, capsys):
        X, y, _ = _make_regression()
        sel = HSICLassoSelector(num_feat=3, **_FIT).fit(X, y)
        result = sel.summary()
        assert isinstance(result, str)
        assert "Rank" in result
        assert "HSIC Score" in result
