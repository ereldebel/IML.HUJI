import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test
from IMLearn.metrics import misclassification_error
from sklearn.metrics import roc_curve

from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
	"""
	Plot the descent path of the gradient descent algorithm

	Parameters:
	-----------
	module: Type[BaseModule]
		Module type for which descent path is plotted

	descent_path: np.ndarray of shape (n_iterations, 2)
		Set of locations if 2D parameter space being the regularization path

	title: str, default=""
		Setting details to add to plot title

	xrange: Tuple[float, float], default=(-1.5, 1.5)
		Plot's x-axis range

	yrange: Tuple[float, float], default=(-1.5, 1.5)
		Plot's x-axis range

	Return:
	-------
	fig: go.Figure
		Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

	Example:
	--------
	fig = plot_descent_path(IMLearn.descent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
	fig.show()
	"""

	def predict_(w):
		return np.array([module(weights=wi).compute_output() for wi in w])

	from utils import decision_surface
	return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
	                                   density=70, showscale=False),
	                  go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1],
	                             mode="markers+lines", marker_color="black")],
	                 layout=go.Layout(xaxis=dict(range=xrange),
	                                  yaxis=dict(range=yrange),
	                                  title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[
	Callable[[], None], List[np.ndarray], List[np.ndarray]]:
	"""
	Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

	Return:
	-------
	callback: Callable[[], None]
		Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
		at each iteration of the algorithm

	values: List[np.ndarray]
		Recorded objective values

	weights: List[np.ndarray]
		Recorded parameters
	"""
	values, weights = [], []

	def callback(value, weight, **kwargs):
		values.append(value)
		weights.append(weight)

	return callback, values, weights


def compare_fixed_learning_rates(
		init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
		etas: Tuple[float] = (1, .1, .01, .001)):
	for module, module_name in ((L1, "L1"), (L2, "L2")):
		best_eta = etas[0]
		best_out = None
		for eta in etas:
			callback, values, weights = get_gd_state_recorder_callback()
			out = GradientDescent(learning_rate=FixedLR(eta),
			                      callback=callback).fit(module(init))
			if best_out is None or module(best_out).compute_output() > module(out).compute_output():
				best_out = out
				best_eta = eta
			plot_descent_path(module, np.array(weights),
			                  fr"$\text{{Descent path of {module_name} with }}\eta = {eta}$").show()
			go.Figure(go.Scatter(x=list(range(1, len(values) + 1)),
			                     y=values,
			                     mode="lines+markers"), layout=go.Layout(
				title=fr"$\text{{Convergence rate of {module_name} with }}\eta = {eta}$")).show()
		print(
			f"minimal value for {module_name} with eta = {best_eta}: {module(best_out).compute_output()} at point {best_out}")


def compare_exponential_decay_rates(
		init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
		eta: float = .1,
		gammas: Tuple[float] = (.9, .95, .99, 1)):
	# Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
	fig = make_subplots(rows=2, cols=2,
	                    subplot_titles=[
		                    fr"$\text{{Convergence rate of L1 with an exponential decay rate with }}\eta = {eta}\text{{ and }}\gamma = {gamma}$"
		                    for gamma in gammas],
	                    horizontal_spacing=.03, vertical_spacing=.06)
	fig.layout.showlegend = False
	for i, gamma in enumerate(gammas):
		callback, values, weights = get_gd_state_recorder_callback()
		best = GradientDescent(learning_rate=ExponentialLR(eta, gamma),
		                       callback=callback, out_type="best").fit(
			L1(init))
		fig.add_traces([go.Scatter(x=list(range(1, len(values) + 1)), y=values,
		                           mode="lines+markers")], rows=(i // 2) + 1,
		               cols=(i % 2) + 1)
		print(
			f"minimal value for L1 with eta = {eta} and gamma = {gamma}: {L1(best).compute_output()} at point {best}")

	# Plot algorithm's convergence for the different values of gamma
	fig.show()

	# Plot descent path for gamma=0.95
	gamma = gammas[1]
	for module, module_name in ((L1, "L1"), (L2, "L2")):
		callback, values, weights = get_gd_state_recorder_callback()
		GradientDescent(learning_rate=ExponentialLR(eta, gamma),
		                callback=callback).fit(module(init))
		plot_descent_path(module, np.array(weights),
		                  fr"$\text{{Descent path of {module_name} with an exponential decay rate with }}\eta = {eta}\text{{ and }}\gamma = {gamma}$").show()


def load_data(path: str = "../datasets/SAheart.data",
              train_portion: float = .8) -> \
		Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
	"""
	Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

	Parameters:
	-----------
	path: str, default= "../datasets/SAheart.data"
		Path to dataset

	train_portion: float, default=0.8
		Portion of dataset to use as a training set

	Return:
	-------
	train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
		Design matrix of train set

	train_y : Series of shape (ceil(train_proportion * n_samples), )
		Responses of training samples

	test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
		Design matrix of test set

	test_y : Series of shape (floor((1-train_proportion) * n_samples), )
		Responses of test samples
	"""
	df = pd.read_csv(path)
	df.famhist = (df.famhist == 'Present').astype(int)
	return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd,
	                        train_portion)


def fit_logistic_regression():
	# Load and split SA Heard Disease dataset
	X_train, y_train, X_test, y_test = load_data()
	X_train, y_train, X_test, y_test = \
		X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

	# Plotting convergence rate of logistic regression over SA heart disease data
	model = LogisticRegression().fit(X_train, y_train)
	y_proba = model.predict_proba(X_train)
	fpr, tpr, thresholds = roc_curve(y_train, y_proba)
	argmax_alpha = np.argmax(tpr - fpr)
	best_alpha = thresholds[argmax_alpha]
	test_error = model.loss(X_test, y_test)
	go.Figure(
		data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
		                 line=dict(color="black", dash='dash'),
		                 name="Random Class Assignment"),
		      go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
		                 name="ROC",
		                 hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}"),
		      go.Scatter(x=[fpr[argmax_alpha]], y=[tpr[argmax_alpha]],
		                 mode='markers', name="maximizer",
		                 marker=dict(color="black",
		                             symbol="x", size=8))],
		layout=go.Layout(
			title=rf"$\text{{ROC Curve of Fitted Model. Best Threshold = }}{best_alpha}\text{{ with test error {test_error}}}$",
			xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
			yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

	# Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
	# of regularization parameter
	for penalty in ("l1", "l2"):

		errors = np.ndarray([100], float)
		evaluation_range = np.linspace(0, 1, 100)
		for i, lam in enumerate(evaluation_range):
			errors[i] = cross_validate(
				LogisticRegression(penalty=penalty, lam=lam), X_train, y_train,
				misclassification_error)[1]
		# go.Figure(
		# 	[go.Scatter(x=evaluation_range, y=errors, mode="lines+markers",
		# 	            name="validation error")]).show()
		minimizer = evaluation_range[np.argmin(errors)]
		loss = LogisticRegression(penalty=penalty, lam=minimizer) \
			.fit(X_train, y_train).loss(X_test, y_test)
		print(
			f"The minimizing lambda for {penalty} was {minimizer} and achieved a misclassification error of {loss}")


if __name__ == '__main__':
	np.random.seed(0)
	compare_fixed_learning_rates()
	compare_exponential_decay_rates()
	fit_logistic_regression()
