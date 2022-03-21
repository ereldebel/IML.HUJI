from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
	# Question 1 - Draw samples and print fitted model
	estimators = UnivariateGaussian(False)
	expectation = 10
	samples = np.random.normal(expectation, 1, size=1000)
	estimators.fit(samples)
	print(f"({estimators.mu_}, {estimators.var_})")

	# Question 2 - Empirically showing sample mean is consistent
	estimated_expectations = []
	number_of_samples = []
	for i in range(10, 1001, 10):
		estimators.fit(samples[:i])
		estimated_expectations.append(abs(estimators.mu_ - expectation))
		number_of_samples.append(i)
	go.Figure(go.Scatter(x=number_of_samples, y=estimated_expectations,
						 mode='markers+lines',
						 name=r'$|\widehat\mu-\mu|$'),
			  layout=go.Layout(
				  title=r"$\text{Distance Between Expectation and Estimation "
						r"of Expectation as Function of Number of Samples}$",
				  xaxis_title=r"$m\text{ - number of samples}$",
				  yaxis_title="r$|\widehat\mu-\mu|$")).show()

	# Question 3 - Plotting Empirical PDF of fitted model
	estimators.fit(samples)
	pdf = estimators.pdf(samples)
	go.Figure(go.Scatter(x=samples, y=pdf,
						 mode='markers',
						 name=r'$\text{Sample Values}$'),
			  layout=go.Layout(
				  title=r"$\text{Sample PDF as Function of Sample Values}$",
				  xaxis_title=r"$\text{Sample values}$",
				  yaxis_title=r"$\text{Sample PDF}$")).show()


def test_multivariate_gaussian():
	# Question 4 - Draw samples and print fitted model
	estimators = MultivariateGaussian()
	mu = [0, 0, 4, 0]
	sigma = np.zeros((4, 4))
	sigma[0, 0] = sigma[2, 2] = sigma[3, 3] = 1
	sigma[1, 1] = 2
	sigma[1, 0] = sigma[0, 1] = 0.2
	sigma[3, 0] = sigma[0, 3] = 0.5
	samples = np.random.multivariate_normal(mu, sigma, size=1000)
	estimators.fit(samples)
	print(estimators.mu_)
	print(estimators.cov_)

	# Question 5 - Likelihood evaluation
	heatmap_size = 200
	f = np.linspace(-10, 10, heatmap_size)
	heatmap = np.ndarray((heatmap_size, heatmap_size))
	for i in range(heatmap_size):
		for j in range(heatmap_size):
			heatmap[i, j] = MultivariateGaussian.log_likelihood(
				np.array((f[i], 0, f[j], 0)), sigma, samples)
	go.Figure(go.Heatmap(z=heatmap, x=f, y=f),
			  layout=go.Layout(
				  title=r"$\text{Heatmap of log-likelihood with }\mu =("
						r"f_\text{1}, 0, f_\text{3}, 0)^\text{T}$",
				  xaxis_title=r"$f_\text{3}\text{ values}$",
				  yaxis_title=r"$f_\text{1}\text{ values}$")).show()

	# Question 6 - Maximum likelihood
	argmax = np.argmax(heatmap)
	print("f_1 estimation: ", "{:.3f}".format(f[(argmax // heatmap_size)]),
		  "f_3 estimation: ", "{:.3f}".format(f[(argmax % heatmap_size)]))


if __name__ == '__main__':
	np.random.seed(0)
	test_univariate_gaussian()
	test_multivariate_gaussian()
