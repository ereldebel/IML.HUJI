from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
	"""
	A decision stump classifier for {-1,1} labels according to the CART algorithm

	Attributes
	----------
	self.threshold_ : float
		The threshold by which the data is split

	self.j_ : int
		The index of the feature by which to split the data

	self.sign_: int
		The label to predict for samples where the value of the j'th feature is about the threshold
	"""

	def __init__(self) -> DecisionStump:
		"""
		Instantiate a Decision stump classifier
		"""
		super().__init__()
		self.threshold_, self.j_, self.sign_ = None, None, None

	def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
		"""
		fits a decision stump to the given data

		Parameters
		----------
		X : ndarray of shape (n_samples, n_features)
			Input data to fit an estimator for

		y : ndarray of shape (n_samples, )
			Responses of input data to fit to
		"""
		thresholds_by_feature = np.ndarray((2, X.shape[1]))
		thr_err = np.ndarray((2, X.shape[1]))
		for j in range(X.shape[1]):
			threshold, error = self._find_threshold(X[:,j], y, 1)
			thresholds_by_feature[1, j] = threshold
			thr_err[1, j] = error
			threshold, error = self._find_threshold(X[:,j], y, -1)
			thresholds_by_feature[0, j] = threshold
			thr_err[0, j] = error

		minimizer = np.unravel_index(np.argmin(thr_err), thr_err.shape)
		self.sign_ = 1 if minimizer[0] == 1 else -1
		self.j_ = minimizer[1]
		self.threshold_ = thresholds_by_feature[minimizer]

	def _predict(self, X: np.ndarray) -> np.ndarray:
		"""
		Predict responses for given samples using fitted estimator

		Parameters
		----------
		X : ndarray of shape (n_samples, n_features)
			Input data to predict responses for

		y : ndarray of shape (n_samples, )
			Responses of input data to fit to

		Returns
		-------
		responses : ndarray of shape (n_samples, )
			Predicted responses of given samples

		Notes
		-----
		Feature values strictly below threshold are predicted as `-sign` whereas values which equal
		to or above the threshold are predicted as `sign`
		"""
		return np.where(X[:, self.j_] < self.threshold_, -self.sign_,
		                self.sign_)

	def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
	                    sign: int) -> Tuple[float, float]:
		"""
		Given a feature vector and labels, find a threshold by which to perform a split
		The threshold is found according to the value minimizing the misclassification
		error along this feature

		Parameters
		----------
		values: ndarray of shape (n_samples,)
			A feature vector to find a splitting threshold for

		labels: ndarray of shape (n_samples,)
			The labels to compare against

		sign: int
			Predicted label assigned to values equal to or above threshold

		Returns
		-------
		thr: float
			Threshold by which to perform split

		thr_err: float between 0 and 1
			Misclassificaiton error of returned threshold

		Notes
		-----
		For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
		which equal to or above the threshold are predicted as `sign`
		"""
		sorted_indices = values.argsort()
		values = values[sorted_indices]
		labels = labels[sorted_indices]
		true_weights = np.where(np.sign(labels) == sign, np.abs(labels), 0)
		false_weights = np.abs(labels) - true_weights
		left_errors = np.concatenate([np.zeros(1), np.cumsum(true_weights)])
		right_errors = np.concatenate(
			[np.cumsum(false_weights[::-1])[::-1], np.zeros(1)])
		errors = left_errors + right_errors
		min_error_index = np.argmin(errors)
		threshold = values[min_error_index - 1] + 0.1 \
			if min_error_index == values.size else values[min_error_index]
		return threshold, errors[min_error_index]

	def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
		"""
		Evaluate performance under misclassification loss function

		Parameters
		----------
		X : ndarray of shape (n_samples, n_features)
			Test samples

		y : ndarray of shape (n_samples, )
			True labels of test samples

		Returns
		-------
		loss : float
			Performance under missclassification loss function
		"""
		return misclassification_error(y, self._predict(X))
