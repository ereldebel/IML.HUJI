import gaussian_estimators as est
import numpy as np


def univariate_gaussian_estimation() -> None:
	estimators = est.UnivariateGaussian(False)
	samples = np.random.normal(10, 1, size=1000)
	estimators.fit(samples)
	print(f"({estimators.mu_}, {estimators.var_})")
	for i in range(10, 1001, 10):
		estimators.fit(samples[:i])


if __name__ == '__main__':
	univariate_gaussian_estimation()
