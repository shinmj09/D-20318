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

# 1. 페이지 설정
st.set_page_config(
    page_title="다이아몬드 가격 예측 분석 웹앱",
    page_icon="💎",
    layout="wide",
)


# 2. 데이터 로드 및 OLS 선형 회귀 모델 학습 (캐싱)
@st.cache_resource
def load_and_train():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    feature_cols = [c for c in train_df.columns if c != "log_price"]

    X_train = train_df[feature_cols].values
    y_train = train_df["log_price"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["log_price"].values

    # 선형 회귀 학습
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # OLS 통계량 직접 계산
    n_samples, n_features = X_train.shape
    X_design = np.column_stack([np.ones(n_samples), X_train])
    y_hat_train = lr.predict(X_train)
    resid_train = y_train - y_hat_train
    dof = n_samples - (n_features + 1)
    mse_train = np.sum(resid_train ** 2) / dof

    cov_beta = mse_train * np.linalg.inv(X_design.T @ X_design)
    std_err = np.sqrt(np.diag(cov_beta))

    params = np.concatenate([[lr.intercept_], lr.coef_])
    param_names = ["const"] + feature_cols
    t_stats = params / std_err
    p_values = np.array([1.0 - erf(abs(t) / sqrt(2.0)) for t in t_stats])

    coef_df = pd.DataFrame(
        {
            "Feature": param_names,
            "Coef": params,
            "Std_Err": std_err,
            "t_stat": t_stats,
            "p_value": p_values,
            "Pct_Impact": (np.exp(params) - 1) * 100,
        }
    )

    # 테스트셋 예측 및 달러($) 역변환
    pred_log = lr.predict(X_test)
    pred_dollar = np.exp(pred_log)
    actual_dollar = np.exp(y_test)

    train_pred_dollar = np.exp(y_hat_train)
    train_actual_dollar = np.exp(y_train)

    metrics = {
        "train_r2_log": r2_score(y_train, y_hat_train),
        "test_r2_log": r2_score(y_test, pred_log),
        "train_r2_dollar": r2_score(train_actual_dollar, train_pred_dollar),
        "test_r2_dollar": r2_score(actual_dollar, pred_dollar),
        "rmse_dollar": np.sqrt(mean_squared_error(actual_dollar, pred_dollar)),
        "mae_dollar": mean_absolute_error(actual_dollar, pred_dollar),
        "mape_dollar": mean_absolute_percentage_error(actual_dollar, pred_dollar) * 100,
    }

    return (
        lr,
        cov_beta,
        mse_train,
        coef_df,
        feature_cols,
        train_df,
        test_df,
        actual_dollar,
        pred_dollar,
        metrics,
    )


try:
    (
        lr,
        cov_beta,
        mse_train,
        coef_df,
        feature_cols,
        train_df,
        test_df,
        actual_dollar,
        pred_dollar,
        metrics,
    ) = load_and_train()
except FileNotFoundError:
    st.error("🚨 GitHub 저장소 루트에 train.csv와 test.csv 파일이 있는지 확인해 주세요.")
    st.stop()


# 3. 헤더 영역
st.title("💎 다이아몬드 가격 예측 선형 회귀 분석 대시보드")
st.markdown(
    "데이터 정제(중복·물리적 오류 제거) → 다중공선성 변수(`x, y, z`) 제거 → "
    "로그 변환(`log_carat`, `log_price`) 및 원-핫 인코딩을 거쳐 학습된 "
    "**OLS 선형 회귀 모델**의 분석 결과와 실시간 가격 예측 시뮬레이터입니다."
)

# 4. 탭 구성
tab1, tab2, tab3 = st.tabs(
    [
        "🔮 실시간 가격 예측 시뮬레이터",
        "📊 모델 성능 및 회귀 계수 분석",
        "🗂️ 상세 데이터 전처리 및 피처 엔지니어링 기록",
    ]
)

# ==============================================================================
# TAB 1: 실시간 가격 예측 시뮬레이터
# ==============================================================================
with tab1:
    st.subheader("다이아몬드 4C 및 비율 스펙 입력")
    col1, col2, col3 = st.columns(3)

    with col1:
        carat = st.slider(
            "무게 (Carat)", min_value=0.20, max_value=5.00, value=1.00, step=0.01
        )
        cut = st.selectbox(
            "컷 등급 (Cut)",
            options=["Fair", "Good", "Very Good", "Premium", "Ideal"],
            index=4,
            help="기준 범주(Reference): Fair",
        )

    with col2:
        color = st.selectbox(
            "색상 등급 (Color, D가 최고등급)",
            options=["D", "E", "F", "G", "H", "I", "J"],
            index=2,
            help="기준 범주(Reference): D",
        )
        clarity = st.selectbox(
            "투명도 등급 (Clarity, IF가 최고등급)",
            options=["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
            index=4,
            help="기준 범주(Reference): I1",
        )

    with col3:
        depth = st.number_input(
            "깊이 비율 (Depth %)", min_value=43.0, max_value=79.0, value=61.8, step=0.1
        )
        table = st.number_input(
            "테이블 비율 (Table %)", min_value=43.0, max_value=95.0, value=57.0, step=0.1
        )

    # 입력값을 모델 피처 형식으로 변환
    input_dict = {col: 0.0 for col in feature_cols}
    input_dict["depth"] = depth
    input_dict["table"] = table
    input_dict["log_carat"] = np.log(carat)

    cut_col = f"cut_{cut}"
    color_col = f"color_{color}"
    clarity_col = f"clarity_{clarity}"

    if cut != "Fair" and cut_col in input_dict:
        input_dict[cut_col] = 1.0
    if color != "D" and color_col in input_dict:
        input_dict[color_col] = 1.0
    if clarity != "I1" and clarity_col in input_dict:
        input_dict[clarity_col] = 1.0

    x_vec = np.array([input_dict[c] for c in feature_cols])
    x_design_vec = np.concatenate([[1.0], x_vec])

    pred_log_val = float(lr.predict(x_vec.reshape(1, -1))[0])
    pred_se = np.sqrt(mse_train + float(x_design_vec @ cov_beta @ x_design_vec.T))
    pi_lower_log = pred_log_val - 1.96 * pred_se
    pi_upper_log = pred_log_val + 1.96 * pred_se

    pred_price = np.
