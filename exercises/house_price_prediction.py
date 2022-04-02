from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def preprocess(data):
    data.dropna(inplace=True)
    data.drop(columns="id", inplace=True)
    data.drop(data[data.date == "0"].index, inplace=True)

    data.date = (pd.Timestamp.now() - pd.to_datetime(data.date, format="%Y%m%dT%H%M%S")).dt.days
    data.yr_renovated = data[["yr_renovated", "yr_built"]].max(axis=1)
    data["sqft_garden"] = data.sqft_lot - data.sqft_living
    data["sqft_garden15"] = data.sqft_lot15 - data.sqft_living15
    zip_code_dummies = pd.get_dummies(data.zipcode, prefix="zipcode")
    design_matrix = pd.concat([data.drop(columns="zipcode"),
                               zip_code_dummies], axis=1)
    return design_matrix


def load_data(filename: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = preprocess(pd.read_csv(filename))
    return data.drop(columns="price"), data.price


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    for feature_name, feature_values in X.iteritems():
        cov_mat = np.cov(feature_values, y)
        corr_feature_y = cov_mat[0, 1] / np.sqrt(cov_mat[0, 0] * cov_mat[1, 1])
        fig = go.Figure(go.Scatter(x=feature_values, y=y, mode="markers"))
        fig.layout = go.Layout(
            title=rf"$\text{{scatter of {feature_name} and {y.name}.}}\rho = {corr_feature_y}$",
            xaxis_title=feature_name, yaxis_title=y.name)
        # if abs(corr_feature_y) < 0.1:
        #     X.drop(columns=feature_name, inplace=True)
        fig.write_html(output_path + f"/{feature_name} evaluation.html")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    design_matrix, labels = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(design_matrix, labels)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(design_matrix, labels)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    model = LinearRegression()
    mean_prediction, prediction_ribbon_radius = \
        np.ndarray([91]), np.ndarray([91])
    p_predictions = np.ndarray([10])
    for p in range(10, 101):
        for i in range(10):
            samples = train_X.sample(frac=(p/100))
            labels = train_y.reindex_like(samples)
            model.fit(samples, labels)
            p_predictions[i] = model.loss(test_X, test_y)
        mean_prediction[p - 10] = np.mean(p_predictions)
        prediction_ribbon_radius[p - 10] = 2 * np.std(p_predictions)
    ps = np.array(range(10, 101))
    fig = go.Figure([go.Scatter(x=ps, y=mean_prediction, mode="markers+lines",
                                name="Mean Prediction"),
                     go.Scatter(x=ps,
                                y=mean_prediction - prediction_ribbon_radius,
                                fill=None, mode="lines",
                                line=dict(color="lightgrey"),
                                name="Confidence Interval"),
                     go.Scatter(x=ps,
                                y=mean_prediction + prediction_ribbon_radius,
                                fill='tonexty', mode="lines",
                                line=dict(color="lightgrey"),
                                showlegend=False)])
    fig.layout = go.Layout(
        title=r"$\mathbb{E}(\text{Loss})\text{ As a Function of }p \text{% "
              r"With a Confidence Interval of } \mathbb{E}(\text{Loss}) \pm"
              r" 2\cdot \sigma_\text{Loss}.$",
        xaxis_title="$p$", yaxis_title=r"$\mathbb{E}(\text{Loss})$")
    fig.show(renderer="browser")
