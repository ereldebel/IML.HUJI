from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import *
from challenge.agoda_cancellation_preprocessor import \
	AgodaCancellationPreprocessor
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd

WEEK = 8


def load_data(filename: str):
	"""
	Load Agoda booking cancellation dataset
	Parameters
	----------
	filename: str
		Path to house prices dataset

	Returns
	-------
	Design matrix and response vector in either of the following formats:
	1) Single dataframe with last column representing the response
	2) Tuple of pandas.DataFrame and Series
	3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
	"""
	full_data = pd.read_csv(filename).drop_duplicates()
	full_data = full_data.drop(
		full_data[full_data["cancellation_policy_code"] == "UNKNOWN"].index)
	full_data["cancellation_datetime"].fillna(0, inplace=True)
	return full_data.dropna()


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray,
                        filename: str):
	"""
	Export to specified file the prediction results of given estimator on given testset.

	File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
	predicted values.

	Parameters
	----------
	estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
		Fitted estimator to use for prediction

	X: ndarray of shape (n_samples, n_features)
		Test design matrix to predict its responses

	filename:
		path to store file at

	"""
	prediction = estimator.predict(X)
	pd.DataFrame(prediction.astype(int),
	             columns=["predicted_values"]).to_csv(filename, index=False)
	print(np.unique(prediction, return_counts=True))


def fill_missing_columns(design_matrix, test_set):
	missing_cols = set(design_matrix.columns) - set(test_set.columns)
	for c in missing_cols:
		test_set[c] = 0
	return test_set[design_matrix.columns]


if __name__ == '__main__':
	np.random.seed(0)

	# Load and preprocess data
	full_data = load_data(
		"../datasets/agoda_cancellation_train.csv")
	p = AgodaCancellationPreprocessor(full_data)
	base_design_matrix, week_specific = p.preprocess(full_data)
	cancellation_labels_list = p.preprocess_labels(
		full_data.cancellation_datetime,
		full_data.booking_datetime)
	design_matrix = pd.DataFrame()
	cancellation_labels = pd.DataFrame()
	for i in range(len(week_specific)):
		pd.concat([design_matrix,
		           pd.concat([base_design_matrix, week_specific[i]], axis=1)])
		pd.concat([cancellation_labels, cancellation_labels_list[i]])
	for i in range(1, WEEK):
		week_data = pd.read_csv(f"week_{i}_test_data.csv")
		test_set_i = p.preprocess(week_data)[0]
		test_set_i_labels = pd.read_csv(f"week_{i}_labels.csv")[
			"cancel"].astype(int)
		p.add_week_features(test_set_i, week_data)
		design_matrix = pd.concat(
			[design_matrix, test_set_i])
		cancellation_labels = pd.concat(
			[cancellation_labels, test_set_i_labels])
		test_set_i = fill_missing_columns(design_matrix, test_set_i)

	design_matrix.fillna(0, inplace=True)
	cancellation_labels = np.array(cancellation_labels).reshape((-1,))
	# Fit model over data
	estimator = RandomForestClassifier(random_state=1, max_leaf_nodes=51,
	                                   n_estimators=200,
	                                   max_depth=15,
	                                   class_weight='balanced_subsample',
	                                   n_jobs=8)
	estimator.fit(design_matrix, cancellation_labels)

	test_set = p.preprocess(pd.read_csv(f"week_{WEEK}_test_data.csv"))[0]
	# Expand test_set with 0 columns to fit the design_matrix shape
	test_set = fill_missing_columns(design_matrix, test_set)

	# Store model predictions over test set
	evaluate_and_export(estimator, test_set,
	                    "319091385_314618794_318839610.csv")
