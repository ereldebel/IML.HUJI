from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        bincount = np.bincount(y.astype(int))
        self.pi_ = bincount / self.classes_.size
        sorted_X = X[y.argsort()]
        grouped_X = np.array(
            np.split(sorted_X, np.cumsum(bincount)[:-1], axis=0), dtype=object)
        self.mu_ = np.array([np.mean(group, axis=0) for group in grouped_X])
        # self.mu_ = np.apply_along_axis(lambda x: np.mean(x, axis=1), 0, grouped_X)
        summed_by_group_outer = np.array(
            np.sum(
                [[(lambda x: np.outer(x, x))(group[j] - self.mu_[i]) for j in
                  range(group.shape[0])] for i, group in enumerate(grouped_X)],
                axis=0), dtype=object)
        self.cov_ = np.sum(summed_by_group_outer, axis=0, dtype="float64") / (
                X.shape[0] - self.classes_.size)
        self._cov_inv = inv(self.cov_)

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
        log_pi_plus_X_cov_mu = np.log(self.pi_) + X @ self._cov_inv @ self.mu_.T
        mu_cov_mu = 0.5 * (self.mu_ @ self._cov_inv @ self.mu_.T).diagonal()
        return np.apply_along_axis(np.argmax, 1, log_pi_plus_X_cov_mu - mu_cov_mu)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")
        constant_factor = 1 / (
                np.power(2 * np.pi, self.mu_.shape[1] / 2) * det(self.cov_))
        X_minus_mu = np.apply_along_axis(lambda x: x - self.mu_, 0, X)
        # creates a 3d column of shape (n_samples, n_features, n_classes)
        exponent = np.apply_along_axis(lambda x_minus_mu: (
                x_minus_mu.T @ self._cov_inv @ x_minus_mu).diagonal(), 0,
                                       X_minus_mu)
        return constant_factor * np.exp(-0.5 * exponent)

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
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
