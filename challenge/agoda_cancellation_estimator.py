from __future__ import annotations

from functools import reduce
from typing import NoReturn, Union
from IMLearn.base import BaseEstimator
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from IMLearn.metrics import mean_square_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, f1_score, recall_score, \
    precision_score, classification_report
from typing import List, Tuple


class AgodaLinearCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self, C, epsilon,
                 threshold: Union[None, float] = None) -> None:
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
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


COMBINED = 0
BAGGING = 1
BOOSTING = 2


class AgodaEnsembleCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self, test_X: np.ndarray, test_y: np.ndarray,
                 n_iter: int = 90, n_estimators_per_type: int = 3,
                 verbose: bool = False,
                 seed_base: int = np.random.randint(1000000)) -> None:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        super().__init__()
        self._n_estimators_per_type = n_estimators_per_type
        self.seed_base = seed_base
        self.verbose = verbose
        self.test_y = test_y
        self.test_X = test_X
        self._n_iter = n_iter
        self._boost_decision_trees: List[DecisionTreeClassifier] = [
            DecisionTreeClassifier(max_depth=i) for i in range(1, 4)]
        self._bagging_decision_trees: List[DecisionTreeClassifier] = [
            DecisionTreeClassifier(max_depth=i) for i in range(800, 1001, 100)]
        self._ada_boost_models: List[Tuple[AdaBoostClassifier, float]] = []
        self._bagging_models: List[Tuple[BaggingClassifier, float]] = []

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
        for i in range(self._n_iter // 9):
            for num_trees in range(30, 51, 10):
                for j in range(3):
                    model = AdaBoostClassifier(self._boost_decision_trees[j],
                                               n_estimators=num_trees,
                                               random_state=self.seed_base + i).fit(
                        X, y)
                    y_pred = model.predict(self.test_X)
                    score = f1_score(self.test_y, y_pred,
                                     labels=[0, 1],
                                     average="macro")
                    self.__add_to_models_if_better_than_worse(model, score,
                                                              self._ada_boost_models,
                                                              "boosting")
            for num_trees in range(10, 31, 10):
                for j in range(3):
                    model = BaggingClassifier(self._bagging_decision_trees[j],
                                              n_jobs=8,
                                              random_state=self.seed_base + i).fit(
                        X, y)
                    y_pred = model.predict(self.test_X)
                    score = f1_score(self.test_y, y_pred,
                                     labels=[0, 1],
                                     average="macro")
                    self.__add_to_models_if_better_than_worse(model, score,
                                                              self._bagging_models,
                                                              "bagging")
        proba = lambda sum, model: sum + model[0].predict_proba(self.test_X)[:, 1]
        boosting_prob = reduce(proba, self._ada_boost_models, 0) / len(
            self._ada_boost_models)
        bagging_prob = reduce(proba, self._bagging_models, 0) / len(
            self._bagging_models)
        best_f1_score = 0
        for bagging_percent in range(101):
            for threshold in range(100):
                bagging_percent /= 100
                threshold /= 100
                y_prob = (bagging_prob * bagging_percent + boosting_prob * (
                        1 - bagging_percent))
                y_pred = (y_prob > threshold).astype(int)
                best_combined_score = f1_score(self.test_y, y_pred,
                                               labels=[0, 1],
                                               average="macro")
                if best_combined_score > best_f1_score:
                    best_f1_score = best_combined_score
                    self._bagging_percent = bagging_percent
                    self._threshold = threshold
                    if self.verbose:
                        print("combined: ", best_combined_score)
        for bagging_percent_eps in range(-50, 51):
            for threshold_eps in range(-50, 51):
                bagging_percent = self._bagging_percent + (
                        bagging_percent_eps / 1000)
                threshold = self._threshold + (threshold_eps / 1000)
                if not 0 <= bagging_percent <= 1 or not 0 < threshold < 1:
                    continue
                y_prob = (bagging_prob * bagging_percent + boosting_prob * (
                        1 - bagging_percent))
                y_pred = (y_prob > threshold).astype(int)
                best_combined_score = f1_score(self.test_y, y_pred,
                                               labels=[0, 1],
                                               average="macro")
                if best_combined_score > best_f1_score:
                    best_f1_score = best_combined_score
                    self._threshold = threshold
                    self._bagging_percent = bagging_percent
                    if self.verbose:
                        print("combined: ", best_combined_score)
        if self.verbose:
            print("combined bagging weight: ", self._bagging_percent)

    def __add_to_models_if_better_than_worse(self, model, score, model_list,
                                             model_type):
        if len(model_list) < self._n_estimators_per_type:
            model_list.append((model, score))
            model_list.sort(key=lambda x: x[1])
            if self.verbose:
                print(model_type, ": ", score)
        elif model_list[0][1] < score:
            model_list[0] = (model, score)
            model_list.sort(key=lambda x: x[1])
            if self.verbose:
                print(model_type, ": ", score)

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
        proba = lambda sum, model: sum + model[0].predict_proba(X)[:, 1]
        boosting_prob = reduce(proba, self._ada_boost_models, 0) / len(
            self._ada_boost_models)
        bagging_prob = reduce(proba, self._bagging_models, 0) / len(
            self._bagging_models)
        y_prob = (bagging_prob * self._bagging_percent + boosting_prob * (
                1 - self._bagging_percent))
        return (y_prob > self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
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
        if self._model_used == BAGGING:
            if self.verbose:
                print("predicted using bagging.")
            return self._bagging_model.predict(X)
        if self._model_used == BOOSTING:
            if self.verbose:
                print("predicted using ada boost.")
            return self._ada_boost_model.predict(X)
        if self.verbose:
            print("predicted with a combined model.")
        boosting_prob = self._ada_boost_model.predict_proba(X)[:, 1]
        bagging_prob = self._bagging_model.predict_proba(X)[:, 1]
        y_prob = (bagging_prob * self._bagging_percent + boosting_prob * (
                1 - self._bagging_percent))
        return np.concat([1 - y_prob, y_prob])

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
