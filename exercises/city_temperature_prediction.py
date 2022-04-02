import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def preprocess(full_data):
    data = full_data[["Country", "City", "Year", "Month", "Temp"]].copy()
    data["DayOfYear"] = (pd.to_datetime(full_data.Date,
                                        dayfirst=True) - pd.to_datetime(
        full_data.Year, format="%Y")).dt.days + 1
    data["Year"] = data["Year"].astype(str)
    return data[data["Temp"] > -72]


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = preprocess(pd.read_csv(filename, parse_dates=True))
    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_data = data[data["Country"] == "Israel"].sort_values("DayOfYear")
    fig = px.scatter(israel_data, x="DayOfYear", y="Temp", color="Year")
    fig.layout = go.Layout(
        title="Temperature in Israel As a Function of the Day of the Year",
        xaxis_title="Day of Year",
        yaxis_title=r"$\text{Temperature }(^\circ C)$")
    fig.show()

    israel_std_by_month = israel_data.groupby("Month")["Temp"].agg("std")
    fig = px.bar(israel_std_by_month, y="Temp",
                 text=israel_std_by_month.apply(lambda x: round(x, 2)))
    fig.layout = go.Layout(
        title="The Standard Deviation of the Temperature in Israel by Month",
        xaxis_title="Month",
        yaxis_title="Standard Deviation")
    fig.show()

    # Question 3 - Exploring differences between countries
    data_mean_and_std_by_month = data.groupby(["Country", "Month"])[
        "Temp"].agg(["mean", "std"]).reset_index()
    fig = px.line(data_mean_and_std_by_month,line_group="Country", x="Month", y="mean", error_y="std", color="Country")
    fig.layout = go.Layout(
        title="The Mean Temperature and It's standard deviation as Error by Country",
        xaxis_title="Month",
        yaxis_title=r"$\text{Mean Temperature }(^\circ C)$")
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(
        israel_data.DayOfYear, israel_data.Temp)
    train = (train_X.values, train_y.values)
    test = (test_X.values, test_y.values)
    losses = []
    for k in range(1, 11):
        poly_model = PolynomialFitting(k)
        loss = round(poly_model.fit(*train).loss(*test), 2)
        print(f"k = {k}, loss = {loss}")
        losses.append(loss)
    fig = go.Figure(go.Bar(y=losses, name="Loss By Degree", text=losses))
    fig.layout = go.Layout(
        title="Loss of Polynomial Fitted to Israel Temperature by Day of Year by Polynomial Degree",
        xaxis_title="Polinomial Degree", yaxis_title="Loss")
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    poly_model = PolynomialFitting(4)
    poly_model.fit(israel_data.DayOfYear.values, israel_data.Temp.values)
    loss_by_country = (data[data.Country != "Israel"].groupby("Country").apply(
        lambda df: poly_model.loss(df.DayOfYear, df.Temp))).reset_index(name="Loss")
    fig = px.bar(loss_by_country, x="Country", y="Loss",
                 text=loss_by_country.Loss.apply(lambda x: round(x, 2)))
    fig.layout = go.Layout(
        title="Loss of Polynomial of 4th Degree Fitted to Israel Temperature by Day of Year Over Different Countries",
        xaxis_title="Country", yaxis_title="Loss")
    fig.show()
