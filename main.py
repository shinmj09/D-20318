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

st.set_page_config(page_title="다이아몬드 가격 예측 웹앱", page_icon="💎", layout="wide")


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
    mse_train = np.sum((y_train - y_hat_train) ** 2) / (n_samples - n_features - 1)

    cov_beta = mse_train * np.linalg.inv(X_design.T @ X_design)
    std_err = np.sqrt(np.diag(cov_beta))

    params = np.concatenate([[lr.intercept_], lr.coef_])
    t_stats = params / std_err
    p_values = np.array([1.0 - erf(abs(t) / sqrt(2.0)) for t in t_stats])

    coef_df = pd.DataFrame({
        "Feature": ["const"] + feature_cols,
        "Coef": params,
        "Std_Err": std_err,
        "t_stat": t_stats,
        "p_value": p_values,
        "Pct_Impact": (np.exp(params) - 1) * 100,
    })

    pred_log = lr.predict(X_test)
    pred_dollar = np.exp(pred_log)
    actual_dollar = np.exp(y_test)

    metrics = {
        "train_r2_log": r2_score(y_train, y_hat_train),
        "test_r2_log": r2_score(y_test, pred_log),
        "train_r2_dollar": r2_score(np.exp(y_train), np.exp(y_hat_train)),
        "test_r2_dollar": r2_score(actual_dollar, pred_dollar),
        "rmse_dollar": np.sqrt(mean_squared_error(actual_dollar, pred_dollar)),
        "mae_dollar": mean_absolute_error(actual_dollar, pred_dollar),
        "mape_dollar": mean_absolute_percentage_error(actual_dollar, pred_dollar) * 100,
    }
    return lr, cov_beta, mse_train, coef_df, feature_cols, train_df, actual_dollar, pred_dollar, metrics


try:
    lr, cov_beta, mse_train, coef_df, feature_cols, train_df, actual_dollar, pred_dollar, metrics = load_and_train()
except FileNotFoundError:
    st.error("🚨 GitHub 저장소 루트에 train.csv와 test.csv 파일이 있는지 확인해 주세요.")
    st.stop()

st.title("💎 다이아몬드 가격 예측 선형 회귀 분석 대시보드")
st.markdown("데이터 정제 → `x, y, z` 제거 → 로그 변환 및 원-핫 인코딩을 거쳐 학습된 **OLS 선형 회귀 모델**입니다.")

tab1, tab2, tab3 = st.tabs([
    "🔮 실시간 가격 예측 시뮬레이터",
    "📊 모델 성능 및 회귀 계수 분석",
    "🗂️ 상세 데이터 전처리 및 피처 엔지니어링 기록",
])

with tab1:
    st.subheader("다이아몬드 4C 및 비율 스펙 입력")
    col1, col2, col3 = st.columns(3)
    with col1:
        carat = st.slider("무게 (Carat)", 0.20, 5.00, 1.00, 0.01)
        cut = st.selectbox("컷 등급 (Cut)", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=4)
    with col2:
        color = st.selectbox("색상 등급 (Color)", ["D", "E", "F", "G", "H", "I", "J"], index=2)
        clarity = st.selectbox("투명도 등급 (Clarity)", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], index=4)
    with col3:
        depth = st.number_input("깊이 비율 (Depth %)", 43.0, 79.0, 61.8, 0.1)
        table = st.number_input("테이블 비율 (Table %)", 43.0, 95.0, 57.0, 0.1)

    input_dict = {col: 0.0 for col in feature_cols}
    input_dict["depth"] = depth
    input_dict["table"] = table
    input_dict["log_carat"] = np.log(carat)

    for prefix, val, ref in [("cut", cut, "Fair"), ("color", color, "D"), ("clarity", clarity, "I1")]:
        col_name = f"{prefix}_{val}"
        if val != ref and col_name in input_dict:
            input_dict[col_name] = 1.0

    x_vec = np.array([input_dict[c] for c in feature_cols])
    x_design_vec = np.concatenate([[1.0], x_vec])

    pred_log_val = float(lr.predict(x_vec.reshape(1, -1))[0])
    pred_se = np.sqrt(mse_train + float(x_design_vec @ cov_beta @ x_design_vec.T))

    pred_price = np.exp(pred_log_val)
    pi_low = np.exp(pred_log_val - 1.96 * pred_se)
    pi_high = np.exp(pred_log_val + 1.96 * pred_se)

    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric("💰 예측 다이아몬드 가격 ($)", f"${pred_price:,.2f}")
    r2.metric("📏 95% 예측 신뢰범위 ($)", f"${pi_low:,.0f} ~${pi_high:,.0f}")
    r3.metric("⚖️ 캐럿당 단가 ($ / ct)", f"${pred_price / carat:,.2f}")

    carat_grid = np.linspace(0.2, 3.5, 80)
    sim_matrix = np.tile(x_vec, (len(carat_grid), 1))
    sim_matrix[:, feature_cols.index("log_carat")] = np.log(carat_grid)
    sim_prices = np.exp(lr.predict(sim_matrix))

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(x=carat_grid, y=sim_prices, mode="lines", name="등급별 가격 곡선"))
    fig_sim.add_trace(go.Scatter(
        x=[carat], y=[pred_price], mode="markers+text", name="현재 스펙",
        text=[f" ${pred_price:,.0f}"], textposition="top left",
        marker=dict(color="red", size=11, symbol="diamond"),
    ))
    fig_sim.update_layout(title="캐럿(Carat) 변화에 따른 예상 가격 추이", xaxis_title="무게 (Carat)", yaxis_title="예측 가격 ($)", height=360)
    st.plotly_chart(fig_sim, use_container_width=True)

with tab2:
    st.subheader("1. 테스트 세트 예측 성능 요약")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Log Scale R²", f"{metrics['test_r2_log']:.5f}", f"Train: {metrics['train_r2_log']:.5f}")
    m2.metric("Dollar Scale R²", f"{metrics['test_r2_dollar']:.5f}", f"Train: {metrics['train_r2_dollar']:.5f}")
    m3.metric("Dollar RMSE", f"${metrics['rmse_dollar']:,.2f}")
    m4.metric("Dollar MAE", f"${metrics['mae_dollar']:,.2f}")
    m5.metric("평균 오차율 (MAPE)", f"{metrics['mape_dollar']:.2f}%")

    c_left, c_right = st.columns(2)
    with c_left:
        idx = np.random.RandomState(42).choice(len(actual_dollar), size=min(2500, len(actual_dollar)), replace=False)
        fig_avp = px.scatter(x=actual_dollar[idx], y=pred_dollar[idx], labels={"x": "실제 가격 ($)", "y": "예측 가격 ($)"}, title="실제 vs 예측 가격", opacity=0.45)
        max_p = max(actual_dollar.max(), pred_dollar.max())
        fig_avp.add_trace(go.Scatter(x=[0, max_p], y=[0, max_p], mode="lines", name="y=x", line=dict(color="red", dash="dash")))
        st.plotly_chart(fig_avp, use_container_width=True)

    with c_right:
        cat_coef = coef_df[coef_df["Feature"].str.startswith(("cut_", "color_", "clarity_"))].copy()
        cat_coef["Group"] = cat_coef["Feature"].apply(lambda x: x.split("_")[0].upper())
        fig_coef = px.bar(cat_coef, x="Pct_Impact", y="Feature", color="Group", orientation="h", title="기준 등급 대비 가격 변화율 (%)")
        fig_coef.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_coef, use_container_width=True)

    st.subheader("2. 전체 회귀 계수 및 통계적 유의성 검정표")
    st.dataframe(
        coef_df.style.format({"Coef": "{:.5f}", "Std_Err": "{:.5f}", "t_stat": "{:.2f}", "p_value": "{:.2e}", "Pct_Impact": "{:+.2f}%"}),
        use_container_width=True,
    )

with tab3:
    st.subheader("1. 데이터 전처리 5단계 파이프라인")
    st.table(pd.DataFrame([
        {"단계": "1단계: 데이터 정제", "처리 내용": "완전 중복 행 제거 / x, y, z 0값 제거
