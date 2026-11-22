import streamlit as st
import pandas as pd
import numpy as np
from math import erf, sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="다이아몬드 가격 예측 웹앱",
    page_icon="💎",
    layout="wide",
)


@st.cache_resource
def load_and_train():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    feature_cols = [c for c in train_df.columns if c != "log_price"]

    X_train = train_df[feature_cols].values
    y_train = train_df["log_price"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["log_price"].values

    lr = LinearRegression().fit(X_train, y_train)

    n_samples, n_features = X_train.shape
    X_design = np.column_stack([np.ones(n_samples), X_train])
    y_hat_train = lr.predict(X_train)
    dof = n_samples - n_features - 1
    mse_train = np.sum((y_train
