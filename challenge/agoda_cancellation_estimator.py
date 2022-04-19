from __future__ import annotations
from typing import NoReturn, Union
from IMLearn.base import BaseEstimator
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from IMLearn.metrics import mean_square_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self, C, epsilon, threshold: Union[None, float] = None) -> None:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        super().__init__()
        self._threshold = threshold
        self._model = make_pipeline(StandardScaler(),
                                    SVR(kernel="rbf", C=C, epsilon=epsilon))
        # self._model = make_pipeline(StandardScaler(),DecisionTreeClassifier(max_depth=number_of_neighbors))
        # self._model = DecisionTreeClassifier(max_depth=1000)


    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        self._model.fit(X, y)

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
        # return np.zeros([X.shape[0]])
        if self._threshold:
            return (self._model.predict(X) > self._threshold).astype(int)
        return self._model.predict(X)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """
        return mean_square_error(y, self.predict(X))
