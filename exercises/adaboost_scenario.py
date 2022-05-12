import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Generate a dataset in R^2 of specified size

	Parameters
	----------
	n: int
		Number of samples to generate

	noise_ratio: float
		Ratio of labels to invert

	Returns
	-------
	X: np.ndarray of shape (n_samples,2)
		Design matrix of samples

	y: np.ndarray of shape (n_samples,)
		Labels of samples
	"""
	'''
	generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
	num_samples: the number of samples to generate
	noise_ratio: invert the label for this ratio of the samples
	'''
	X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
	y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
	y[np.random.choice(n, int(noise_ratio * n))] *= -1
	return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
	(train_X, train_y), (test_X, test_y) = \
		generate_data(train_size, noise), generate_data(test_size, noise)

	# Question 1: Train- and test errors of AdaBoost in noiseless case
	model = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
	learners_range = range(1, n_learners + 1)
	train_loss = np.array([model.partial_loss(train_X, train_y, t) for t in
	                       learners_range])
	test_loss = np.array([model.partial_loss(test_X, test_y, t) for t in
	                      learners_range])
	title = "Missclassification Loss of Adaboost with Prediction" \
	        " Stump as a Function of the Number of Learners"
	title = title if noise == 0 else title + f". Noise: {noise}"
	go.Figure([go.Scatter(x=list(learners_range), y=train_loss,
	                      mode="markers+lines",
	                      name="prediction loss over train set"),
	           go.Scatter(x=list(learners_range), y=test_loss,
	                      mode="markers+lines",
	                      name="prediction loss over test set")],
	          layout=go.Layout(
		          title=title,
		          xaxis={"rangemode": "tozero", "title": "Number of Learners"},
		          yaxis={"rangemode": "tozero",
		                 "title": "Misclassification Loss"}
	          )).show()

	# Question 2: Plotting decision surfaces
	T = [5, 50, 100, 250]
	lims = np.array([np.r_[train_X, test_X].min(axis=0),
	                 np.r_[train_X, test_X].max(axis=0)]).T + np.array(
		[-.1, .1])

	symbols = np.array(["x", "circle"])
	if noise == 0:
		fig = make_subplots(rows=2, cols=2,
		                    subplot_titles=[f"{t} Learners:" for t in T],
		                    horizontal_spacing=0.01, vertical_spacing=.03)
		for i, t in enumerate(T):
			fig.add_traces(
				[decision_surface(lambda x: model.partial_predict(x, t),
				                  lims[0], lims[1], showscale=False),
				 go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
				            mode="markers", showlegend=False,
				            marker=dict(color=test_y, symbol=symbols[
					            ((test_y + 1) / 2).astype(int)],
				                        colorscale=[custom[0],
				                                    custom[-1]],
				                        line=dict(color="black",
				                                  width=1)))],
				rows=(i // 2) + 1, cols=(i % 2) + 1)

		fig.update_layout(
			title="Decision Boundaries of Adaboost with Prediction Stump "
			      "for Different Number of Learners",
			margin=dict(t=100)) \
			.update_xaxes(visible=False).update_yaxes(visible=False)
		fig.show()

		# Question 3: Decision surface of best performing ensemble
		best_t = int(np.argmin(test_loss))
		fig = go.Figure([decision_surface(
			lambda x: model.partial_predict(x, best_t), lims[0],
			lims[1], showscale=False),
			go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
			           mode="markers", showlegend=False,
			           marker=dict(color=train_y, symbol=symbols[
				           ((train_y + 1) / 2).astype(int)],
			                       colorscale=[custom[0], custom[-1]],
			                       line=dict(color="black", width=0.2)))],
			layout=go.Layout(
				title=f"Decision Boundaries of Adaboost with Prediction Stump "
				      f"with {best_t + 1} learners. Accuracy: "
				      f"{1 - test_loss[best_t]}",
				margin=dict(t=100))) \
			.update_xaxes(visible=False).update_yaxes(visible=False)
		fig.show()

	# Question 4: Decision surface with weighted samples
	D = model.D_ / np.max(model.D_) * 8
	title = rf"$\text{{Decision Boundaries of Adaboost with Prediction "\
	        rf"Stump with {n_learners} learners with marker size by }}"\
	        rf"\mathcal{{D}}^{{(T)}}\text{{ weights}}$"
	title = title if noise == 0 else title[:-3] + rf". Noise: {noise}}}$"
	fig = go.Figure([decision_surface(
		lambda x: model.partial_predict(x, n_learners), lims[0],
		lims[1], showscale=False),
		go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
		           mode="markers", showlegend=False,
		           marker=dict(color=D, size=D, symbol=symbols[
			           ((train_y + 1) / 2).astype(int)],
		                       colorscale=[custom[0], custom[-1]],
		                       line=dict(color="black", width=0.2)))],
		layout=go.Layout(
			title=title,
			margin=dict(t=100))) \
		.update_xaxes(visible=False).update_yaxes(visible=False)
	fig.show()


if __name__ == '__main__':
	np.random.seed(0)
	fit_and_evaluate_adaboost(0)
	fit_and_evaluate_adaboost(0.4)
