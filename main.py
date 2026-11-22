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
    cols = [c for c in train_df.columns if c != "log_price"]

    X_tr = train_df[cols].values
    y_tr = train_df["log_price"].values
    X_te = test_df[cols].values
    y_te = test_df["log_price"].values

    lr = LinearRegression()
    lr.fit(X_tr, y_tr)

    n_rows, n_cols = X_tr.shape
    ones = np.ones(n_rows)
    X_des = np.column_stack([ones, X_tr])
    y_hat = lr.predict(X_tr)
    err = y_tr - y_hat
    dof = n_rows - n_cols - 1
    mse = float(np.dot(err, err) / dof)

    xtx_inv = np.linalg.inv(X_des.T @ X_des)
    cov_b = mse * xtx_inv
    se_b = np.sqrt(np.diag(cov_b))

    params = np.concatenate([[lr.intercept_], lr.coef_])
    t_vals = params / se_b
    p_vals = np.array(
        [1.0 - erf(abs(t) / sqrt(2.0)) for t in t_vals]
    )

    coef_df = pd.DataFrame(
        {
            "Feature": ["const"] + cols,
            "Coef": params,
            "Std_Err": se_b,
            "t_stat": t_vals,
            "p_value": p_vals,
            "Pct_Impact": (np.exp(params) - 1.0) * 100.0,
        }
    )

    pred_log = lr.predict(X_te)
    pred_usd = np.exp(pred_log)
    act_usd = np.exp(y_te)

    metrics = {
        "tr_r2_log": r2_score(y_tr, y_hat),
        "te_r2_log": r2_score(y_te, pred_log),
        "tr_r2_usd": r2_score(np.exp(y_tr), np.exp(y_hat)),
        "te_r2_usd": r2_score(act_usd, pred_usd),
        "rmse": np.sqrt(mean_squared_error(act_usd, pred_usd)),
        "mae": mean_absolute_error(act_usd, pred_usd),
        "mape": mean_absolute_percentage_error(
            act_usd, pred_usd
        )
        * 100.0,
    }
    return (
        lr,
        cov_b,
        mse,
        coef_df,
        cols,
        train_df,
        act_usd,
        pred_usd,
        metrics,
    )


try:
    (
        lr,
        cov_b,
        mse,
        coef_df,
        cols,
        train_df,
        act_usd,
        pred_usd,
        metrics,
    ) = load_and_train()
except FileNotFoundError:
    st.error("🚨 train.csv 및 test.csv 파일을 확인해 주세요.")
    st.stop()

st.title("💎 다이아몬드 가격 예측 선형 회귀 분석 대시보드")
st.markdown(
    "데이터 정제, `x/y/z` 제거, 로그 변환, 원-핫 인코딩을 "
    "거쳐 학습된 **OLS 선형 회귀 모델**입니다."
)

tab1, tab2, tab3 = st.tabs(
    [
        "🔮 실시간 가격 예측 시뮬레이터",
        "📊 모델 성능 및 회귀 계수 분석",
        "🗂️ 상세 데이터 전처리 기록",
    ]
)

with tab1:
    st.subheader("다이아몬드 4C 및 비율 스펙 입력")
    c1, c2, c3 = st.columns(3)
    with c1:
        carat = st.slider(
            "무게 (Carat)", 0.20, 5.00, 1.00, 0.01
        )
        cut = st.selectbox(
            "컷 등급 (Cut)",
            ["Fair", "Good", "Very Good", "Premium", "Ideal"],
            index=4,
        )
    with c2:
        color = st.selectbox(
            "색상 등급 (Color)",
            ["D", "E", "F", "G", "H", "I", "J"],
            index=2,
        )
        clarity = st.selectbox(
            "투명도 등급 (Clarity)",
            [
                "I1",
                "SI2",
                "SI1",
                "VS2",
                "VS1",
                "VVS2",
                "VVS1",
                "IF",
            ],
            index=4,
        )
    with c3:
        depth = st.number_input(
            "깊이 비율 (Depth %)", 43.0, 79.0, 61.8, 0.1
        )
        table = st.number_input(
            "테이블 비율 (Table %)", 43.0, 95.0, 57.0, 0.1
        )

    inp = {c: 0.0 for c in cols}
    inp["depth"] = depth
    inp["table"] = table
    inp["log_carat"] = np.log(carat)

    pairs = [
        ("cut", cut, "Fair"),
        ("color", color, "D"),
        ("clarity", clarity, "I1"),
    ]
    for prefix, val, ref in pairs:
        key = f"{prefix}_{val}"
        if val != ref and key in inp:
            inp[key] = 1.0

    x_v = np.array([inp[c] for c in cols])
    x_des_v = np.concatenate([[1.0], x_v])

    p_log = float(lr.predict(x_v.reshape(1, -1))[0])
    var_pred = mse + float(x_des_v @ cov_b @ x_des_v.T)
    p_se = np.sqrt(var_pred)

    price = np.exp(p_log)
    pi_lo = np.exp(p_log - 1.96 * p_se)
    pi_hi = np.exp(p_log + 1.96 * p_se)

    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric("💰 예측 가격 ($)", f"${price:,.2f}")
    r2.metric(
        "📏 95% 예측 신뢰범위 ($)",
        f"${pi_lo:,.0f} ~${pi_hi:,.0f}",
    )
    r3.metric(
        "⚖️ 캐럿당 단가 ($ / ct)",
