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

    # OLS 통계량 (표준오차, t-통계량, p-값, 예측 공분산 행렬) 직접 계산
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
# TAB 3: 상세 데이터 전처리 및 피처 엔지니어링 기록
# ==============================================================================
with tab3:
    st.subheader("1. 데이터 전처리 전체 파이프라인 개요")
    st.markdown(
        """
        선형 회귀 모델의 기본 가정(**선형성, 다중공선성 배제, 오차항의 등분산성**)을 충족하고 예측 정확도를 극대화하기 위해 
        원본 `diamonds` 데이터에 **총 5단계의 체계적인 전처리 및 피처 엔지니어링**을 적용했습니다.
        """
    )

    pipeline_df = pd.DataFrame(
        [
            {
                "단계": "1단계: 데이터 정제 (Cleaning)",
                "세부 처리 작업": "① 완전 중복 행 제거  ② 물리적 치수(x, y, z) 0값 제거  ③ 극단적 치수 입력 이상치(y=58.9, 31.8 / z=31.8) 제거",
                "적용 사유 및 분석 근거": "동일 관측치 중복 학습을 방지하고, 부피가 0이거나 현실적으로 불가능한 크기의 입력 오류 레코드가 회귀선을 왜곡하는 현상 차단",
            },
            {
                "단계": "2단계: 다중공선성 변수 제거",
                "세부 처리 작업": "물리적 길이·너비·깊이 변수인 `x`, `y`, `z` 열 삭제",
                "적용 사유 및 분석 근거": "다이아몬드 무게(`carat`)는 부피(`x × y × z`)와 밀접하게 비례하므로 함께 투입 시 분산팽창요인(VIF)이 급등하여 회귀 계수 부호 역전 및 표준오차 팽창 유발",
            },
            {
                "단계": "3단계: 비선형성 및 이분산성 해소",
                "세부 처리 작업": "독립변수 `carat`과 종속변수 `price`에 자연로그(`np.log`) 변환 적용 (`log_carat`, `log_price`)",
                "적용 사유
