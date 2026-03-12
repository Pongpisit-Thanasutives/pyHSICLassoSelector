"""
HSICLassoSelector — core implementation.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.linear_model._base import LinearModel
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class HSICLassoSelector(LinearModel, RegressorMixin, TransformerMixin, BaseEstimator):
    """
    Scikit-learn compatible feature selector using HSIC Lasso.

    HSIC Lasso (Hilbert-Schmidt Independence Criterion Lasso) identifies
    features that are maximally dependent on the target while being
    minimally redundant with each other — making it a powerful,
    kernel-based alternative to linear filter methods (e.g. ANOVA,
    mutual information) and greedy wrappers (e.g. RFE).

    It wraps `pyHSICLasso <https://github.com/riken-aip/pyHSICLasso>`_
    and exposes a fully sklearn-compatible estimator interface, including
    ``Pipeline``, ``GridSearchCV``, ``ColumnTransformer``, and
    ``set_output(transform="pandas")`` support.

    Parameters
    ----------
    num_feat : int, default=5
        Number of features to select.

    B : int, default=0
        Block size for block HSIC Lasso.
        ``0`` uses the full (exact) HSIC Lasso; values like ``20``–``100``
        enable the scalable block approximation for large datasets.

    M : int, default=1
        Number of block permutations.  Only relevant when ``B > 0``.
        Higher values improve stability at the cost of runtime.

    task : {"regression", "classification"}, default="regression"
        Determines which HSIC Lasso objective is used internally.
        Use ``"classification"`` for discrete / categorical targets.

    n_jobs : int, default=-1
        Number of parallel threads used by pyHSICLasso internally.
        ``-1`` uses all available cores (pyHSICLasso default).

    discrete_x : bool, default=False
        When ``True``, uses a Delta kernel for all input features
        (appropriate for discrete / categorical feature matrices).

    max_neighbors : int, default=10
        Maximum neighbours for the k-NN neighbour graph used when
        summarising related features (pyHSICLasso default is 10).

    covars : np.ndarray, default=np.array([])
        Optional covariate matrix to condition on (shape
        ``(n_covariates, n_samples)``).  Pass an empty array to disable
        (pyHSICLasso default).

    covars_kernel : {"Gaussian", "Delta"}, default="Gaussian"
        Kernel type applied to the covariate matrix.

    feature_names : array-like of str or None, default=None
        Optional feature names forwarded to pyHSICLasso.  When ``None``
        names are auto-generated as ``"x0"``, ``"x1"``, …

    random_state : int or None, default=None
        Seed for ``numpy.random`` to make results reproducible.
        Note: pyHSICLasso uses numpy global random state internally.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features_in_,)
        Full-length coefficient vector from pyHSICLasso (beta), zero for
        unselected features.  Follows the sklearn convention used by
        ``Lasso``, ``ElasticNet``, etc.

    scores_ : ndarray of shape (n_features_selected_,)
        HSIC scores for the selected features, in the same order as
        ``support_indices_``.  Analogous to ``scores_`` on
        ``SelectKBest`` / ``SelectFpr``.

    support_indices_ : ndarray of shape (n_features_selected_,)
        Integer column indices of the selected features.  Equivalent to
        ``get_support(indices=True)``.

    feature_importances_ : ndarray of shape (n_features_in_,)
        Full-length importance vector: HSIC score at selected positions,
        zero elsewhere.  Follows the sklearn tree-estimator convention.

    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    n_features_selected_ : int
        Actual number of features selected (≤ ``num_feat``).

    feature_names_in_ : ndarray of str, optional
        Feature names seen during :meth:`fit` (set when ``feature_names``
        is supplied or when the input is a pandas DataFrame).

    hsic_lasso_ : pyHSICLasso.HSICLasso
        The raw fitted pyHSICLasso object for advanced use
        (``hl.dump()``, ``hl.plot_path()``, etc.).

    Examples
    --------
    Basic regression usage:

    >>> import numpy as np
    >>> from hsic_lasso_selector import HSICLassoSelector
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((300, 30))
    >>> y = 3 * X[:, 0] - 2 * X[:, 5] + rng.standard_normal(300) * 0.1
    >>> sel = HSICLassoSelector(num_feat=3, B=50, random_state=0)
    >>> sel.fit(X, y)
    HSICLassoSelector(B=50, num_feat=3, random_state=0)
    >>> sel.selected_indices_
    array([0, 5, ...])
    >>> sel.transform(X).shape
    (300, 3)

    Inside a Pipeline:

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import Ridge
    >>> pipe = Pipeline([
    ...     ("sel", HSICLassoSelector(num_feat=10, B=50)),
    ...     ("reg", Ridge()),
    ... ])
    >>> pipe.fit(X, y)
    Pipeline(...)

    Notes
    -----
    pyHSICLasso must be installed separately::

        pip install pyHSICLasso

    References
    ----------
    Yamada, M. et al. (2014). High-dimensional Feature Selection by
    Feature-Wise Kernelized Lasso. *Neural Computation*, 26(1), 185–207.
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        num_feat: int = 5,
        B: int = 20,
        M: int = 3,
        task: str = "regression",
        n_jobs: int = -1,
        discrete_x: bool = False,
        max_neighbors: int = 10,
        covars=None,
        covars_kernel: str = "Gaussian",
        feature_names=None,
        random_state=None,
    ) -> None:
        self.num_feat = num_feat
        self.B = B
        self.M = M
        self.task = task
        self.n_jobs = n_jobs
        self.discrete_x = discrete_x
        self.max_neighbors = max_neighbors
        self.covars = covars
        self.covars_kernel = covars_kernel
        self.feature_names = feature_names
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _resolve_feat_names(self, n_features: int) -> list[str]:
        """Return a feature-name list of the correct length."""
        if self.feature_names is not None:
            names = list(self.feature_names)
            if len(names) != n_features:
                raise ValueError(
                    f"feature_names has {len(names)} entries but X has "
                    f"{n_features} features."
                )
            return names
        # Try names stored by sklearn (e.g. from a DataFrame)
        if hasattr(self, "feature_names_in_"):
            return list(self.feature_names_in_)
        return [f"x{i}" for i in range(n_features)]

    def _resolve_covars(self):
        """Return a valid covars array (empty ndarray when not specified)."""
        if self.covars is None:
            return np.array([])
        return np.asarray(self.covars)

    @staticmethod
    def _import_pyHSICLasso():
        try:
            from pyHSICLasso import HSICLasso  # noqa: N813
            return HSICLasso
        except ImportError as exc:
            raise ImportError(
                "pyHSICLasso is required but not installed.\n"
                "Install it with:  pip install pyHSICLasso"
            ) from exc

    @staticmethod
    def _densify_beta(beta, n_features: int) -> np.ndarray:
        """
        Convert any beta representation from pyHSICLasso to a plain
        1-D float64 C-contiguous ndarray of length ``n_features``.

        pyHSICLasso stores beta as a list/array of per-feature arrays,
        e.g. beta[i] = array([score_i]).  Indexing this with numpy gives
        dtype('O') which breaks any downstream numpy casting.

        We also handle sparse matrices, np.matrix, 2-D arrays, and plain lists.
        """
        import scipy.sparse as _sp

        # Case 1: sparse matrix → dense
        if _sp.issparse(beta):
            beta = beta.toarray()

        # Case 2: np.matrix → plain ndarray
        if isinstance(beta, np.matrix):
            beta = np.asarray(beta)

        # Case 3: list or object array whose elements are arrays/scalars
        # (the real pyHSICLasso case: beta[i] = array([val]))
        if isinstance(beta, (list, tuple)):
            out = np.zeros(n_features, dtype=np.float64)
            for i, v in enumerate(beta):
                if i >= n_features:
                    break
                try:
                    out[i] = float(np.asarray(v).ravel()[0])
                except (IndexError, TypeError, ValueError):
                    out[i] = 0.0
            return out

        beta = np.asarray(beta)

        # Case 4: object array whose elements are arrays (after np.asarray)
        if beta.dtype == object:
            out = np.zeros(n_features, dtype=np.float64)
            flat = beta.ravel()
            for i, v in enumerate(flat):
                if i >= n_features:
                    break
                try:
                    out[i] = float(np.asarray(v).ravel()[0])
                except (IndexError, TypeError, ValueError):
                    out[i] = 0.0
            return out

        # Case 5: normal numeric array — flatten, cast, pad/trim
        beta = beta.ravel()
        if not np.issubdtype(beta.dtype, np.floating):
            beta = beta.astype(np.float64)
        else:
            beta = beta.astype(np.float64, copy=False)

        beta = np.ascontiguousarray(beta, dtype=np.float64)

        if beta.shape[0] != n_features:
            padded = np.zeros(n_features, dtype=np.float64)
            padded[: min(beta.shape[0], n_features)] = beta[:n_features]
            beta = padded

        return beta

    # ------------------------------------------------------------------ #
    #  Core sklearn API                                                    #
    # ------------------------------------------------------------------ #

    def fit(self, X, y):
        """
        Fit the HSIC Lasso selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,) or (n_samples, 1)
            Target vector.

        Returns
        -------
        self : HSICLassoSelector
            Fitted estimator (returns ``self`` for chaining).
        """
        HSICLasso = self._import_pyHSICLasso()

        # Validate parameters
        if self.task not in {"regression", "classification"}:
            raise ValueError(
                f"task must be 'regression' or 'classification', got {self.task!r}."
            )
        if self.num_feat < 1:
            raise ValueError(f"num_feat must be ≥ 1, got {self.num_feat}.")
        if self.B < 0:
            raise ValueError(f"B must be ≥ 0, got {self.B}.")

        # Input validation
        y_numeric = self.task == "regression"
        X, y = check_X_y(X, y, ensure_2d=True, y_numeric=y_numeric)
        y = y.ravel()

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        feat_names = self._resolve_feat_names(n_features)

        # Clamp num_feat to available features
        effective_num_feat = min(self.num_feat, n_features)
        if effective_num_feat < self.num_feat:
            import warnings
            warnings.warn(
                f"num_feat={self.num_feat} exceeds n_features={n_features}; "
                f"clamped to {effective_num_feat}.",
                UserWarning,
                stacklevel=2,
            )

        # Seed for reproducibility — random_state may be an int, a
        # numpy RandomState, a numpy Generator, or None.
        # hidimstat passes a numpy RandomState object directly.
        # np.random.seed() only accepts int|None, so we must handle all cases.
        if self.random_state is not None:
            if isinstance(self.random_state, np.random.RandomState):
                # Extract a fresh integer seed from the RandomState
                np.random.seed(self.random_state.randint(0, 2**31 - 1))
            elif isinstance(self.random_state, np.random.Generator):
                np.random.seed(int(self.random_state.integers(0, 2**31 - 1)))
            else:
                np.random.seed(int(self.random_state))

        hl = HSICLasso()
        hl.input(X, y, featname=feat_names)

        run_kwargs = dict(
            num_feat=effective_num_feat,
            B=self.B,
            M=self.M,
            discrete_x=self.discrete_x,
            max_neighbors=self.max_neighbors,
            n_jobs=self.n_jobs,
            covars=self._resolve_covars(),
            covars_kernel=self.covars_kernel,
        )

        if self.task == "regression":
            hl.regression(**run_kwargs)
        else:
            hl.classification(**run_kwargs)

        # Store results — sklearn-conventional attribute names
        self.hsic_lasso_ = hl

        # Force concrete dtypes — pyHSICLasso may return object arrays,
        # sparse matrices, or mixed-type lists; joblib workers will fail
        # with dtype('O') when numpy tries to index with these.
        raw_idx = hl.get_index()
        raw_scores = hl.get_index_score()

        self.support_indices_ = np.array(
            [int(i) for i in raw_idx], dtype=np.intp
        )
        self.scores_ = np.array(
            [float(s) for s in raw_scores], dtype=np.float64
        )
        self.n_features_selected_ = int(len(self.support_indices_))

        # coef_: hl.beta may be a scipy sparse matrix, np.matrix, or a
        # 2-D array.  We need a 1-D, C-contiguous, float64 ndarray with
        # no object scalars — hidimstat indexes into it inside joblib workers.
        self.coef_ = self._densify_beta(hl.beta, n_features)

        # intercept_: HSIC Lasso has no bias term; set to 0.0 so that
        # shap.LinearExplainer and other linear-model consumers work out of the box.
        self.intercept_ = 0.0

        # feature_importances_: full-length, HSIC score at selected positions
        importances = np.zeros(n_features, dtype=np.float64)
        importances[self.support_indices_] = self.scores_
        self.feature_importances_ = importances

        return self

    def transform(self, X):
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_selected : ndarray of shape (n_samples, num_feat)
        """
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features; fitted on {self.n_features_in_}."
            )
        idx = self.support_indices_.astype(np.intp, copy=False)
        return X[:, idx]

    # ------------------------------------------------------------------ #
    #  Additional sklearn-compatible methods                               #
    # ------------------------------------------------------------------ #

    def get_support(self, indices: bool = False):
        """
        Return a mask or index array of the selected features.

        Parameters
        ----------
        indices : bool, default=False
            If ``True`` return an integer index array; otherwise a
            boolean mask of length ``n_features_in_``.

        Returns
        -------
        support : ndarray of bool or int
        """
        check_is_fitted(self)
        if indices:
            return self.support_indices_.astype(np.intp, copy=False)
        mask = np.zeros(self.n_features_in_, dtype=np.bool_)
        mask[self.support_indices_] = True
        return mask

    def get_feature_names_out(self, input_features=None):
        """
        Return feature names for the selected features.

        Compatible with sklearn ≥ 1.0's ``set_output`` API.

        Parameters
        ----------
        input_features : array-like of str or None, default=None

        Returns
        -------
        feature_names_out : ndarray of str, shape (num_feat,)
        """
        check_is_fitted(self)
        if input_features is not None:
            names = np.asarray(input_features, dtype=object)
        elif self.feature_names is not None:
            names = np.asarray(self.feature_names, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            names = self.feature_names_in_
        else:
            names = np.array(
                [f"x{i}" for i in range(self.n_features_in_)], dtype=object
            )
        return names[self.support_indices_]

    def _more_tags(self):
        return {"requires_y": True, "no_validation": False}

    def __getstate__(self):
        """
        Custom pickle state — exclude ``hsic_lasso_`` from serialisation.

        ``hsic_lasso_`` holds the raw pyHSICLasso object which may contain
        object-dtype arrays that break numpy casting inside joblib workers.
        All information needed by sklearn (``coef_``, ``scores_``,
        ``support_indices_``, ``feature_importances_``) is already stored
        as clean float64 / intp arrays and is preserved.
        """
        state = self.__dict__.copy()
        state.pop("hsic_lasso_", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # hsic_lasso_ is not restored; set to None so callers can check
        if "hsic_lasso_" not in self.__dict__:
            self.hsic_lasso_ = None

    # ------------------------------------------------------------------ #
    #  Convenience / diagnostics                                           #
    # ------------------------------------------------------------------ #

    def summary(self, input_features=None) -> str:
        """
        Return a formatted ranking table of selected features.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Feature names to display.  Falls back to ``feature_names``
            set at construction or auto-generated names.

        Returns
        -------
        table : str
            Human-readable table (also printed to stdout).

        Examples
        --------
        >>> sel.summary()
        Rank   Feature                        HSIC Score
        -------------------------------------------------------
        1      gene_BRCA1                       0.183421
        2      gene_TP53                        0.141209
        ...
        """
        check_is_fitted(self)
        names = self.get_feature_names_out(input_features)
        order = np.argsort(self.scores_)[::-1]

        header = f"{'Rank':<6} {'Feature':<35} {'HSIC Score':>12}"
        sep = "-" * len(header)
        rows = [header, sep]
        for rank, idx in enumerate(order, 1):
            rows.append(
                f"{rank:<6} {str(names[idx]):<35} {self.scores_[idx]:>12.6f}"
            )
        table = "\n".join(rows)
        print(table)
        return table
