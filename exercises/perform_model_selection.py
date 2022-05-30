from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
	RidgeRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

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
	noise_vector = np.random.normal(0, noise, n_samples)
	X = np.linspace(-1.2, 2, n_samples)
	y = f(X) + noise_vector
	noiseless_y = f(X)
	train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=2 / 3)
	fig = go.Figure([go.Scatter(x=X, y=noiseless_y, mode="lines+markers",
	                            name="True model"),
	                 go.Scatter(x=train_X, y=train_y,
	                            mode="markers",
	                            name="Train samples"),
	                 go.Scatter(x=test_X, y=test_y, mode="markers",
	                            name="Test samples")], layout=go.Layout(
		title=rf"$\text{{True polynomial and noisy samples. noise = {noise},"
		      rf" m = {n_samples} samples}}$"))
	fig.show()

	# Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
	errors = np.ndarray([11, 2], float)
	degree_range = range(11)
	for degree in degree_range:
		errors[degree] = cross_validate(PolynomialFitting(degree), train_X,
		                                train_y,
		                                mean_square_error)
	minimizer = np.argmin(errors[:, 1])
	fig = go.Figure(
		[go.Scatter(x=list(degree_range), y=errors[:, 0], mode="lines+markers",
		            name="train error"),
		 go.Scatter(x=list(degree_range), y=errors[:, 1], mode="lines+markers",
		            name="validation error"),
		 go.Scatter(x=[minimizer], y=[errors[minimizer, 1]], mode="markers",
		            name="validation error minimizer")
		 ], layout=go.Layout(
			title=rf"$\text{{Mean Train and Validation Errors Using 5-fold"
			      rf" Cross Validation. noise = {noise}, m = {n_samples} "
			      rf"samples}}$"))
	fig.show()

	# Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
	model = PolynomialFitting(int(minimizer)).fit(train_X, train_y)
	print(f"with noise {noise} and {n_samples} samples, k* was {minimizer} and"
	      f" got a {model.loss(test_X, test_y)} test error.")


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
	select_polynomial_degree(noise=0)
	select_polynomial_degree(n_samples=1500, noise=10)
