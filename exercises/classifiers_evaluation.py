from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px

pio.templates.default = "simple_white"
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback(p: Perceptron, xn: np.ndarray, yn: int):
            losses.append(p._loss(X, y))

        perceptron = Perceptron(callback=callback)
        perceptron.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = px.line(y=losses, x=np.array(range(len(losses))))
        fig.layout = go.Layout(
            title=f"Perceptron training loss as a function of the training iteration.\nClasses are {n}.",
            xaxis={"title": "Iteration"},
            yaxis={"rangemode": "tozero", "title": "Loss"})
        fig.show(renderer="browser")


def get_ellipse(mu: np.ndarray, cov: np.ndarray, draw_legend: bool = True):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black", showlegend=draw_legend,
                      name="covariance")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    colors = {0: "seagreen", 1: "green", 2: "salmon"}
    symbols = {0: "star-square", 1: "hexagram", 2: "star-triangle-up"}
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        lda = LDA().fit(X, y)
        naive_bayes = GaussianNaiveBayes().fit(X, y)
        lda_predictions = lda.predict(X)
        naive_bayes_predictions = naive_bayes.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig = make_subplots(cols=2, subplot_titles=(
            f"LDA. Accuracy = {round(accuracy(lda_predictions, y), 2)}",
            f"Gaussian Naive Bayes. Accuracy = {round(accuracy(naive_bayes_predictions, y), 2)}"),
                            x_title="", y_title="")

        for n, model, predictions in [
            (1, lda, lda_predictions),
            (2, naive_bayes, naive_bayes_predictions)]:
            for true_cls in model.classes_:
                for pred_cls in model.classes_:
                    filtered_X = X[(predictions == pred_cls) & (y == true_cls)]
                    # Add traces for data-points setting symbols and colors
                    fig.add_trace(
                        go.Scatter(x=filtered_X[:, 0], y=filtered_X[:, 1],
                                   mode="markers",
                                   marker=dict(
                                       color="light" + colors[pred_cls],
                                       symbol=symbols[true_cls],
                                       size=10,
                                       line=dict(
                                           color=colors[pred_cls],
                                           width=2)),
                                   showlegend=bool(n - 1),
                                   name=f"true: {int(true_cls)}, predicted: {int(pred_cls)}"),
                        row=1, col=n)
            # Add `X` dots specifying fitted Gaussians' means
            fig.add_trace(go.Scatter(x=model.mu_[:, 0], y=model.mu_[:, 1],
                                     mode="markers",
                                     marker=dict(color="black",
                                                 symbol="x", size=10),
                                     showlegend=bool(n - 1), name="mean"),
                          row=1, col=n)
            # Add ellipses depicting the covariances of the fitted Gaussians
            for i, mu in enumerate(model.mu_):
                cov = model.cov_ if n == 1 else np.diag(
                    naive_bayes.vars_[i, :])
                fig.add_trace(get_ellipse(mu, cov, not bool(n + i - 1)), row=1,
                              col=n)
        fig.update_layout(title_text=f)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
