from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
	RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
	"""
	Simulate data from a polynomial model and use cross-validation to select the best fitting degree

	Parameters
	----------
	n_samples: int, default=100
		Number of samples to generate

	noise: float, default = 5
		Noise level to simulate in responses
	"""
	# Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
	# and split into training- and testing portions
	f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
	base_samples = np.linspace(-1.2, 2, n_samples)
	noise = np.random.normal(0, noise, n_samples)
	X = f(base_samples) + noise
	y = f(base_samples)
	X_dataframe = pd.DataFrame(X).set_index(base_samples)
	train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X),
	                                                    pd.DataFrame(y), 2 / 3)
	train_X_index = train_X.index.to_numpy().reshape(-1)
	train_X = train_X.values.reshape(-1)
	test_X_index = test_X.index.to_numpy().reshape(-1)
	test_X = test_X.values.reshape(-1)
	test_y = test_y.values.reshape(-1)
	train_y = train_y.values.reshape(-1)
	fig = go.Figure([go.Scatter(x=base_samples, y=y, mode="lines+markers",
	                            name="True model"),
	                 go.Scatter(x=train_X_index, y=train_X, mode="markers",
	                            name="Train samples"),
	                 go.Scatter(x=test_X_index, y=test_X, mode="markers",
	                            name="Test samples")], layout=go.Layout(
		title=r"$\text{True polynomial and noisy samples}$"))
	# fig.show()

	# Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
	errors = np.ndarray([11, 2],float)
	for degree in range(11):
		errors[degree] = cross_validate(PolynomialFitting(degree), train_X, train_y,
		                                mean_square_error)
	print(errors)
	fig = go.Figure([go.Scatter(x=list(range(11)),y=errors[:,0]),go.Scatter(x=list(range(11)),y=errors[:,1])])
	fig.show()

	# Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
	raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
	"""
	Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
	values for Ridge and Lasso regressions

	Parameters
	----------
	n_samples: int, default=50
		Number of samples to generate

	n_evaluations: int, default = 500
		Number of regularization parameter values to evaluate for each of the algorithms
	"""
	# Question 6 - Load diabetes dataset and split into training and testing portions
	raise NotImplementedError()

	# Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
	raise NotImplementedError()

	# Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
	raise NotImplementedError()


if __name__ == '__main__':
	np.random.seed(0)
	select_polynomial_degree()
