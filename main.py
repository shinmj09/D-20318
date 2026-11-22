import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 페이지 설정
st.set_page_config(
    page_title="다이아몬드 가격 예측 분석 웹앱",
    page_icon="💎",
    layout="wide",
)


# 2. 데이터 로드 및 모델 학습 (캐싱)
@st.cache_resource
def load_and_train():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    feature_cols = [c for c in train_df.columns if c != "log_price"]

    X_train = sm.add_constant(train_df[feature_cols], has_constant="add")
    y_train = train_df["log_price"]

    X_test = sm.add_constant(test_df[feature_cols], has_constant="add")
    y_test = test_df["log_price"]

    # OLS 선형 회귀 학습
    model = sm.OLS(y_train, X_train).fit()

    # 테스트셋 예측 및 달러($) 역변환
    pred_log = model.predict(X_test)
    pred_dollar = np.exp(pred_log)
    actual_dollar = np.exp(y_test)

    # 학습셋 성능 (과적합 점검용)
    train_pred_log = model.predict(X_train)
    train_pred_dollar = np.exp(train_pred_log)
    train_actual_dollar = np.exp(y_train)

    metrics = {
        "train_r2_log": r2_score(y_train, train_pred_log),
        "test_r2_log": r2_score(y_test, pred_log),
        "train_r2_dollar": r2_score(train_actual_dollar, train_pred_dollar),
        "test_r2_dollar": r2_score(actual_dollar, pred_dollar),
        "rmse_dollar": np.sqrt(mean_squared_error(actual_dollar, pred_dollar)),
        "mae_dollar": mean_absolute_error(actual_dollar, pred_dollar),
        "mape_dollar": mean_absolute_percentage_error(actual_dollar, pred_dollar) * 100,
    }

    return (
        model,
        feature_cols,
        train_df,
        test_df,
        X_test,
        y_test,
        pred_log,
        actual_dollar,
        pred_dollar,
        metrics,
    )


try:
    (
        model,
        feature_cols,
        train_df,
        test_df,
        X_test,
        y_test,
        pred_log,
        actual_dollar,
        pred_dollar,
        metrics,
    ) = load_and_train()
except FileNotFoundError:
    st.error("🚨 GitHub 저장소 루트 경로에 `train.csv`와 `test.csv` 파일이 있는지 확인해 주세요.")
    st.stop()


# 3. 헤더 영역
st.title("💎 다이아몬드 가격 예측 선형 회귀 분석 대시보드")
st.markdown(
    """
    데이터 정제(중복·물리적 오류 제거) → 다중공선성 변수(`x, y, z`) 제거 → 로그 변환(`log_carat`, `log_price`) 및 
    원-핫 인코딩을 거쳐 학습된 **OLS 선형 회귀 모델**의 분석 결과와 실시간 가격 예측 시뮬레이터입니다.
    """
)

# 4. 탭 구성
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔮 실시간 가격 예측 시뮬레이터",
        "📊 모델 성능 및 회귀 계수 분석",
        "📉 잔차 진단 (Residual Diagnostics)",
        "🗂️ 데이터 및 전처리 파이프라인 요약",
    ]
)

# ==============================================================================
# TAB 1: 실시간 가격 예측 시뮬레이터
# ==============================================================================
with tab1:
    st.subheader("다이아몬드 4C 및 비율 스펙 입력")
    col1, col2, col3 = st.columns(3)

    with col1:
        carat = st.slider("무게 (Carat)", min_value=0.20, max_value=5.00, value=1.00, step=0.01)
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
        depth = st.number_input("깊이 비율 (Depth %)", min_value=43.0, max_value=79.0, value=61.8, step=0.1)
        table = st.number_input("테이블 비율 (Table %)", min_value=43.0, max_value=95.0, value=57.0, step=0.1)

    # 입력값을 모델 피처 형식으로 변환
    input_dict = {col: 0.0 for col in feature_cols}
    input_dict["depth"] = depth
    input_dict["table"] = table
    input_dict["log_carat"] = np.log(carat)

    if cut != "Fair" and f"cut_{cut}" in input_dict:
        input_dict[f"cut_{cut}"] = 1.0
    if color != "D" and f"color_{color}" in input_dict:
        input_dict[f"color_{color}"] = 1.0
    if clarity != "I1" and f"clarity_{clarity}" in input_dict:
        input_dict[f"clarity_{clarity}"] = 1.0

    input_df = pd.DataFrame([input_dict])[feature_cols]
    input_sm = sm.add_constant(input_df, has_constant="add")

    # 예측 및 95% 예측 구간(Prediction Interval) 산출
    pred_summary = model.get_prediction(input_sm).summary_frame(alpha=0.05)
    pred_log_val = pred_summary["mean"].iloc[0]
    pi_lower_log = pred_summary["obs_ci_lower"].iloc[0]
    pi_upper_log = pred_summary["obs_ci_upper"].iloc[0]

    pred_price = np.exp(pred_log_val)
    pi_lower_price = np.exp(pi_lower_log)
    pi_upper_price = np.exp(pi_upper_log)
    price_per_carat = pred_price / carat

    st.divider()
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("💰 예측 다이아몬드 가격 ($)", f"${pred_price:,.2f}")
    res_col2.metric("📏 95% 예측 신뢰범위 ($)", f"${pi_lower_price:,.0f} ~${pi_upper_price:,.0f}")
    res_col3.metric("⚖️ 캐럿당 단가 ($ / ct)", f"${price_per_carat:,.2f}")

    # 캐럿 변화에 따른 가격 곡선 시뮬레이션
    carat_grid = np.linspace(0.2, 3.5, 80)
    sim_rows = []
    for c_val in carat_grid:
        row = input_dict.copy()
        row["log_carat"] = np.log(c_val)
        sim_rows.append(row)
    sim_df = sm.add_constant(pd.DataFrame(sim_rows)[feature_cols], has_constant="add")
    sim_prices = np.exp(model.predict(sim_df))

    fig_sim = go.Figure()
    fig_sim.add_trace(
        go.Scatter(
            x=carat_grid,
            y=sim_prices,
            mode="lines",
            name=f"선택 등급 곡선 ({cut} / {color} / {clarity})",
            line=dict(color="#1f77b4", width=3),
        )
    )
    fig_sim.add_trace(
        go.Scatter(
            x=[carat],
            y=[pred_price],
            mode="markers+text",
            name="현재 선택 스펙",
            text=[f" ${pred_price:,.0f}"],
            textposition="top left",
            marker=dict(color="#d62728", size=12, symbol="diamond"),
        )
    )
    fig_sim.update_layout(
        title="동일 등급 조건에서 캐럿(Carat) 증가에 따른 예상 가격 추이",
        xaxis_title="무게 (Carat)",
        yaxis_title="예측 가격 ($)",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_sim, use_container_width=True)


# ==============================================================================
# TAB 2: 모델 성능 및 회귀 계수 분석
# ==============================================================================
with tab2:
    st.subheader("1. 테스트 세트 예측 성능 요약")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Log Scale R²", f"{metrics['test_r2_log']:.5f}", f"Train: {metrics['train_r2_log']:.5f}")
    m2.metric("Dollar Scale R²", f"{metrics['test_r2_dollar']:.5f}", f"Train: {metrics['train_r2_dollar']:.5f}")
    m3.metric("Dollar RMSE", f"${metrics['rmse_dollar']:,.2f}")
    m4.metric("Dollar MAE", f"${metrics['mae_dollar']:,.2f}")
    m5.metric("평균 오차율 (MAPE)", f"{metrics['mape_dollar']:.2f}%")

    col_left, col_right = st.columns(2)

    with col_left:
        # 실제 vs 예측 산점도
        sample_idx = np.random.RandomState(42).choice(
            len(actual_dollar), size=min(2500, len(actual_dollar)), replace=False
        )
        fig_avp = px.scatter(
            x=actual_dollar.iloc
