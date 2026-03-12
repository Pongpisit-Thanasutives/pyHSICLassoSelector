"""
hsic-lasso-selector
===================

A scikit-learn compatible feature selector based on HSIC Lasso.

Quick start
-----------
>>> from hsic_lasso_selector import HSICLassoSelector
>>> sel = HSICLassoSelector(num_feat=10, B=50, task="regression")
>>> sel.fit(X_train, y_train)
>>> X_selected = sel.transform(X_train)
"""

from importlib.metadata import version, PackageNotFoundError

from ._selector import HSICLassoSelector

try:
    __version__ = version("hsic-lasso-selector")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.0.0.dev0"

__all__ = ["HSICLassoSelector", "__version__"]
