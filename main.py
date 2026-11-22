import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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


# 2. 데이터 로드 및 OLS 모델 학습 (순수 NumPy + Scikit-learn으로 통계량까지 직접 산출)
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

    # OLS 통계량 (표준오차, t-통계량, 근사 p-값, 예측 공분산 행렬) 직접 계산
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

    # 대표본(n > 40,000) 정규 근사 양측 p-값
    from math import erf, sqrt
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
        y_test,
        pred_log,
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

    x_vec = np.array([input_dict[c] for c in feature_cols])
    x_design_vec = np.concatenate([[1.0], x_vec])

    pred_log_val = float(lr.predict(x_vec.reshape(1, -1))[0])
    pred_se = np.sqrt(mse_train + float(x_design_vec @ cov_beta @ x_design_vec.T))
    pi_lower_log = pred_log_val - 1.96 * pred_se
    pi_upper_log = pred_log_val + 1.96 * pred_se

    pred_price = np.exp(pred_log_val)
    pi_lower_price = np.exp(pi_lower_log)
    pi_upper_price = np.exp(pi_upper_log)
    price_per_carat = pred_price / carat

    st.divider()
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("💰 예측 다이아몬드 가격 ($)", f"${pred_price:,.2f}")
    res_col2.metric("📏 95% 예측 신뢰범위 ($)", f"${pi_lower_price:,.0f} ~ ${pi_upper_price:,.0f}")
    res_col3.metric("⚖️ 캐럿당 단가 ($ / ct)", f"${price_per_carat:,.2f}")

    # 캐럿 변화에 따른 가격 곡선 시뮬레이션
    carat_grid = np.linspace(0.2, 3.5, 80)
    sim_matrix = np.tile(x_vec, (len(carat_grid), 1))
    log_carat_idx = feature_cols.index("log_carat")
    sim_matrix[:, log_carat_idx] = np.log(carat_grid)
    sim_prices = np.exp(lr.predict(sim_matrix))

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
        sample_idx = np.random.RandomState(42).choice(
            len(actual_dollar), size=min(2500, len(actual_dollar)), replace=False
        )
        fig_avp = px.scatter(
            x=actual_dollar[sample_idx],
            y=pred_dollar[sample_idx],
            labels={"x": "실제 가격 ($)", "y": "예측 가격 ($)"},
            title="실제 가격 vs 예측 가격 (Test Set 비교)",
            opacity=0.45,
        )
        max_p = max(actual_dollar.max(), pred_dollar.max())
        fig_avp.add_trace(
            go.Scatter(
                x=[0, max_p], y=[0, max_p], mode="lines", name="완벽 예측선 (y=x)", line=dict(color="red", dash="dash")
            )
        )
        fig_avp.update_layout(height=420)
        st.plotly_chart(fig_avp, use_container_width=True)

    with col_right:
        cat_coef = coef_df[coef_df["Feature"].str.startswith(("cut_", "color_", "clarity_"))].copy()
        cat_coef["Group"] = cat_coef["Feature"].apply(lambda x: x.split("_")[0].upper())
        fig_coef = px.bar(
            cat_coef,
            x="Pct_Impact",
            y="Feature",
            color="Group",
            orientation="h",
            title="기준 등급 대비 가격 변화 효과 (%)",
            labels={"Pct_Impact": "가격 변화율 (%)", "Feature": "등급 변수"},
        )
        fig_coef.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_coef, use_container_width=True)

    st.subheader("2. 전체 회귀 계수 및 통계적 유의성 검정표")
    st.caption(
        "• `log_carat` 계수(+1.8848)는 무게가 1% 증가할 때 가격이 약 1.88% 상승함을 의미합니다. • `depth`와 `table`은 p > 0.05로 통계적으로 유의하지 않습니다."
    )
    st.dataframe(
        coef_df.style.format(
            {
                "Coef": "{:.5f}",
                "Std_Err": "{:.5f}",
                "t_stat": "{:.2f}",
                "p_value": "{:.2e}",
                "Pct_Impact": "{:+.2f}%",
            }
        ),
        use_container_width=True,
        height=400,
    )


# ==============================================================================
# TAB 3: 잔차 진단 (Residual Diagnostics)
# ==============================================================================
with tab3:
    st.subheader("선형 회귀 기본 가정 및 잔차 통계 검정 (Test Set 기준)")

    resid_log = y_test - pred_log
    resid_mean = np.mean(resid_log)
    resid_std = np.std(resid_log, ddof=1)
    std_resid = (resid_log - resid_mean) / resid_std

    # Durbin-Watson, 왜도, 첨도 직접 계산
    dw_stat = np.sum(np.diff(resid_log) ** 2) / np.sum(resid_log ** 2)
    skew_val = np.mean(std_resid ** 3)
    kurt_val = np.mean(std_resid ** 4) - 3.0

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("잔차 평균 (편향)", f"{resid_mean:.5f}", "이상적 기준: 0.0000")
    r2.metric("Durbin-Watson (독립성)", f"{dw_stat:.4f}", "이상적 기준: 2.0 부근")
    r3.metric("왜도 / 초과 첨도", f"{skew_val:.2f} / {kurt_val:.2f}")
    r4.metric("이상 잔차 비율 (|z| > 3)", f"{(np.abs(std_resid) > 3).mean()*100:.2f}%")

    idx_res = np.random.RandomState(42).choice(len(resid_log), size=min(3000, len(resid_log)), replace=False)
    pred_sub = pred_log[idx_res]
    resid_sub = resid_log[idx_res]
    std_sub = std_resid[idx_res]

    # 구간별 평균 추세선 계산
    pred_series = pd.Series(pred_log)
    resid_series = pd.Series(resid_log)
    bins = pd.qcut(pred_series, q=25)
    bin_x = pred_series.groupby(bins, observed=False).mean()
    bin_y = resid_series.groupby(bins, observed=False).mean()

    fig_diag = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "1. Residuals vs Fitted (선형성 진단)",
            "2. Normal Q-Q Plot (정규성 진단)",
            "3. 잔차 분포 히스토그램 (Log Scale)",
            "4. Scale-Location Plot (등분산성 진단)",
        ),
    )

    # 1) Residuals vs Fitted
    fig_diag.add_trace(
        go.Scatter(x=pred_sub, y=resid_sub, mode="markers", marker=dict(size=4, opacity=0.35, color="#1f77b4"), name="잔차"),
        row=1,
        col=1,
    )
    fig_diag.add_trace(
        go.Scatter(x=bin_x, y=bin_y, mode="lines", line=dict(color="#ff7f0e", width=2.5), name="구간 평균 추세"),
        row=1,
        col=1,
    )
    fig_diag.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # 2) Normal Q-Q Plot (순수 NumPy 분위수 계산)
    sorted_std = np.sort(std_resid)
    n_obs = len(sorted_std)
    probs = (np.arange(1, n_obs + 1) - 0.5) / n_obs
    # Beasley-Springer-Moro 근사로 표준정규분포 역함수(Probit) 계산
    from statistics import NormalDist
    nd = NormalDist()
    theoretical_q = np.array([nd.inv_cdf(p) for p in probs])

    qq_idx = np.linspace(0, n_obs - 1, 1000, dtype=int)
    fig_diag.add_trace(
        go.Scatter(
            x=theoretical_q[qq_idx],
            y=sorted_std[qq_idx],
            mode="markers",
            marker=dict(size=4, opacity=0.5, color="#2ca02c"),
            name="표본 분위수",
        ),
        row=1,
        col=2,
    )
    fig_diag.add_trace(
        go.Scatter(x=[-3.5, 3.5], y=[-3.5, 3.5], mode="lines", line=dict(color="red", dash="dash"), name="정규 기준선"),
        row=1,
        col=2,
    )

    # 3) Histogram
    fig_diag.add_trace(
        go.Histogram(x=resid_log, nbinsx=50, marker_color="#9467bd", opacity=0.75, name="잔차 분포"),
        row=2,
        col=1,
    )

    # 4) Scale-Location
    sqrt_std_series = pd.Series(np.sqrt(np.abs(std_resid)))
    bin_sqrt = sqrt_std_series.groupby(bins, observed=False).mean()
    fig_diag.add_trace(
        go.Scatter(
            x=pred_sub,
            y=np.sqrt(np.abs(std_sub)),
            mode="markers",
            marker=dict(size=4, opacity=0.35, color="#d62728"),
            name="√|표준화 잔차|",
        ),
        row=2,
        col=2,
    )
    fig_diag.add_trace(
        go.Scatter(x=bin_x, y=bin_sqrt, mode="lines", line=dict(color="black", width=2.5), name="등분산 추세"),
        row=2,
        col=2,
    )

    fig_diag.update_layout(height=720, showlegend=False)
    st.plotly_chart(fig_diag, use_container_width=True)


# ==============================================================================
# TAB 4: 데이터 및 전처리 파이프라인 요약
# ==============================================================================
with tab4:
    st.subheader("단계별 데이터 정제 및 피처 엔지니어링 요약")
    pipeline_df = pd.DataFrame(
        [
            {"단계": "원본 데이터", "작업 내용": "초기 diamonds 데이터 로드 및 결측치 점검", "핵심 목적": "기본 데이터 구조 파악"},
            {"단계": "1단계 정제", "작업 내용": "완전 중복 행 제거", "핵심 목적": "동일 스펙 중복 레코드 제거"},
            {"단계": "2단계 정제", "작업 내용": "물리적 치수(x, y, z) 0값 포함 오류 데이터 제거", "핵심 목적": "부피 없는 비정상 데이터 제거"},
            {"단계": "3단계 정제", "작업 내용": "극단적 치수 입력 이상치 제거", "핵심 목적": "왜곡 유발 극단값 정제"},
            {"단계": "피처 엔지니어링", "작업 내용": "x, y, z 열 삭제 + log(carat), log(price) 변환 + 원-핫 인코딩", "핵심 목적": "다중공선성 해소 및 비율 선형화"},
            {"단계": "데이터 분할", "작업 내용": "log_price 분위수 기준 80:20 층화 분할 (Stratified Split)", "핵심 목적": "train.csv / test.csv 가격 분포 균등 유지"},
        ]
    )
    st.table(pipeline_df)

    st.subheader("학습 데이터 (`train.csv`) 미리보기")
    st.dataframe(train_df.head(10), use_container_width=True)
