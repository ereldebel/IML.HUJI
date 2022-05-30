from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from IMLearn.metrics import mean_square_error


class RidgeRegression(BaseEstimator):
	"""
	Ridge Regression Estimator

	Solving Ridge Regression optimization problem
	"""

	def __init__(self, lam: float,
	             include_intercept: bool = True) -> RidgeRegression:
		"""
		Initialize a ridge regression model

		Parameters
		----------
		lam: float
			Regularization parameter to be used when fitting a model

		include_intercept: bool, default=True
			Should fitted model include an intercept or not

		Attributes
		----------
		include_intercept_: bool
			Should fitted model include an intercept or not

		coefs_: ndarray of shape (n_features,) or (n_features+1,)
			Coefficients vector fitted by linear regression. To be set in
			`LinearRegression.fit` function.
		"""

		"""
		Initialize a ridge regression model
		:param lam: scalar value of regularization parameter
		"""
		super().__init__()
		self.coefs_ = None
		self.include_intercept_ = include_intercept
		self.lam_ = lam

	def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
		"""
		Fit Ridge regression model to given samples

		Parameters
		----------
		X : ndarray of shape (n_samples, n_features)
			Input data to fit an estimator for

		y : ndarray of shape (n_samples, )
			Responses of input data to fit to

		Notes
		-----
		Fits model with or without an intercept depending on value of `self.include_intercept_`
		"""
		if self.include_intercept_:
			X = X - X.mean(axis=0)
			y = y - y.mean()
		v, vec_sigma, u = np.linalg.svd(X)
		vec_sigma_l = np.apply_along_axis(
			lambda x: x / (x * x + self.lam_), 0, vec_sigma)
		sigma_l = np.zeros((v.shape[0], u.shape[0]))
		sigma_l[:vec_sigma_l.size, :vec_sigma_l.size] = np.diag(
			vec_sigma_l)
		self.coefs_ = v @ sigma_l @ u.T @ y

	def _predict(self, X: np.ndarray) -> np.ndarray:
		"""
		Predict responses for given samples using fitted estimator

		Parameters
		----------
		X : ndarray of shape (n_samples, n_features)
			Input data to predict responses for

		Returns
		-------
		responses : ndarray of shape (n_samples, )
			Predicted responses of given samples
		"""
		return X @ self.coefs_

	def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
		"""
		Evaluate performance under MSE loss function

		Parameters
		----------
		X : ndarray of shape (n_samples, n_features)
			Test samples

		y : ndarray of shape (n_samples, )
			True labels of test samples

		Returns
		-------
		loss : float
			Performance under MSE loss function
		"""
		return mean_square_error(y,self.predict(X))
