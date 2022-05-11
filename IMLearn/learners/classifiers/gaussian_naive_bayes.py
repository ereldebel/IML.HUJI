from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, bincount = np.unique(y, return_counts=True)
        self.pi_ = bincount / y.size
        sorted_X = X[y.argsort()]
        grouped_X = np.array(
            np.split(sorted_X, np.cumsum(bincount)[:-1], axis=0), dtype=object)
        self.mu_ = np.array([np.mean(group, axis=0) for group in grouped_X])
        self.vars_ = np.array(
            [np.var(group, axis=0, ddof=1) for group in grouped_X])

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
        log_factor = np.log(self.pi_) - 0.5 * self.mu_.shape[1] * np.log(
            (2 * np.pi)) - np.sum(np.log(self.vars_), axis=1)
        exponent = np.apply_along_axis(
            lambda x: np.sum((x - self.mu_) ** 2 / self.vars_, axis=1), 1, X)
        return np.apply_along_axis(np.argmax, 1, log_factor - 0.5 * exponent)

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
                np.power(2 * np.pi, self.mu_.shape[1] / 2) * np.prod(
            self.vars_, axis=1))
        exponent = np.apply_along_axis(
            lambda x: np.sum((x - self.mu_) ** 2 / self.vars_, axis=1), 1, X)
        return np.apply_along_axis(lambda x: constant_factor * x, 1,
                                   np.exp(-0.5 * exponent))

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
