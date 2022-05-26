from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
	"""
	Evaluate metric by cross-validation for given estimator

	Parameters
	----------
	estimator: BaseEstimator
		Initialized estimator to use for fitting the data

	X: ndarray of shape (n_samples, n_features)
	   Input data to fit

	y: ndarray of shape (n_samples, )
	   Responses of input data to fit to

	scoring: Callable[[np.ndarray, np.ndarray, ...], float]
		Callable to use for evaluating the performance of the cross-validated model.
		When called, the scoring function receives the true- and predicted values for each sample
		and potentially additional arguments. The function returns the score for given input.

	cv: int
		Specify the number of folds.

	Returns
	-------
	train_score: float
		Average train score over folds

	validation_score: float
		Average validation score over folds
	"""
	X_parts = np.array_split(X, cv)
	y_parts = np.array_split(y, cv)
	train_sum, validation_sum = 0, 0
	for k in range(cv):
		X_k_fold = np.concatenate(
			[part for j, part in enumerate(X_parts) if k != j])
		y_k_fold = np.concatenate(
			[part for j, part in enumerate(y_parts) if k != j])
		estimator.fit(X_k_fold, y_k_fold)
		train_sum += scoring(y_k_fold, estimator.predict(X_k_fold))
		validation_sum += scoring(y_parts[k], estimator.predict(X_parts[k]))
	return train_sum / cv, validation_sum / cv
