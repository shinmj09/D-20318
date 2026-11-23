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
    p_vals = np.array([1.0 - erf(abs(t) / sqrt(2.0)) for t in t_vals])

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
        "mape": mean_absolute_percentage_error(act_usd, pred_usd) * 100.0,
    }
    return lr, cov_b, mse, coef_df, cols, train_df, act_usd, pred_usd, metrics


try:
    lr, cov_b, mse, coef_df, cols, train_df, act_usd, pred_usd, metrics = load_and_train()
except FileNotFoundError:
    st.error("🚨 GitHub 저장소 루트에 train.csv 및 test.csv 파일이 있는지 확인해 주세요.")
    st.stop()

st.title("💎 다이아몬드 가격 예측 선형 회귀 분석 대시보드")
st.markdown(
    "데이터 정제, `x/y/z` 제거, 로그 변환, 원-핫 인코딩을 거쳐 학습된 **OLS 선형 회귀 모델**입니다."
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
        carat = st.slider("무게 (Carat)", 0.20, 5.00, 1.00, 0.01)
        cut = st.selectbox("컷 등급 (Cut)", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=4)
    with col2 if False else c2:
        color = st.selectbox("색상 등급 (Color)", ["D", "E", "F", "G", "H", "I", "J"], index=2)
        clarity = st.selectbox(
            "투명도 등급 (Clarity)",
            ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
            index=4,
        )
    with c3:
        depth = st.number_input("깊이 비율 (Depth %)", 43.0, 79.0, 61.8, 0.1)
        table = st.number_input("테이블 비율 (Table %)", 43.0, 95.0, 57.0, 0.1)

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
    unit_price = price / carat

    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric("💰 예측 가격 ($)", f"${price:,.2f}")
    r2.metric("📏 95% 예측 신뢰범위 ($)", f"${pi_lo:,.0f} ~ ${pi_hi:,.0f}")
    r3.metric("⚖️ 캐럿당 단가 ($ / ct)", f"${unit_price:,.2f}")

    grid = np.linspace(0.2, 3.5, 80)
    sim_x = np.tile(x_v, (len(grid), 1))
    lc_idx = cols.index("log_carat")
    sim_x[:, lc_idx] = np.log(grid)
    sim_y = np.exp(lr.predict(sim_x))

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=grid, y=sim_y, mode="lines", name="등급별 가격 곡선"))
    fig1.add_trace(
        go.Scatter(
            x=[carat],
            y=[price],
            mode="markers+text",
            name="현재 스펙",
            text=[f" ${price:,.0f}"],
            textposition="top left",
            marker=dict(color="red", size=11, symbol="diamond"),
        )
    )
    fig1.update_layout(
        title="캐럿(Carat) 변화에 따른 예상 가격 추이",
        xaxis_title="무게 (Carat)",
        yaxis_title="예측 가격 ($)",
        height=360,
    )
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.subheader("1. 테스트 세트 예측 성능 요약")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Log Scale R²", f"{metrics['te_r2_log']:.5f}", f"Train: {metrics['tr_r2_log']:.5f}")
    m2.metric("Dollar Scale R²", f"{metrics['te_r2_usd']:.5f}", f"Train: {metrics['tr_r2_usd']:.5f}")
    m3.metric("Dollar RMSE", f"${metrics['rmse']:,.2f}")
    m4.metric("Dollar MAE", f"${metrics['mae']:,.2f}")
    m5.metric("평균 오차율 (MAPE)", f"{metrics['mape']:.2f}%")

    cl, cr = st.columns(2)
    with cl:
        rng = np.random.RandomState(42)
        n_sub = min(2500, len(act_usd))
        idx = rng.choice(len(act_usd), size=n_sub, replace=False)
        fig2 = px.scatter(
            x=act_usd[idx],
            y=pred_usd[idx],
            labels={"x": "실제 ($)", "y": "예측 ($)"},
            title="실제 vs 예측 가격",
            opacity=0.45,
        )
        hi_p = float(max(act_usd.max(), pred_usd.max()))
        fig2.add_trace(
            go.Scatter(
                x=[0, hi_p],
                y=[0, hi_p],
                mode="lines",
                name="y=x",
                line=dict(color="red", dash="dash"),
            )
        )
        st.plotly_chart(fig2, use_container_width=True)

    with cr:
        prefixes = ("cut_", "color_", "clarity_")
        mask = coef_df["Feature"].str.startswith(prefixes)
        cat_df = coef_df[mask].copy()
        cat_df["Group"] = cat_df["Feature"].apply(lambda s: s.split("_")[0].upper())
        fig3 = px.bar(
            cat_df,
            x="Pct_Impact",
            y="Feature",
            color="Group",
            orientation="h",
            title="기준 등급 대비 가격 변화율 (%)",
        )
        fig3.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("2. 전체 회귀 계수 및 통계 검정표")
    fmt_dict = {
        "Coef": "{:.5f}",
        "Std_Err": "{:.5f}",
        "t_stat": "{:.2f}",
        "p_value": "{:.2e}",
        "Pct_Impact": "{:+.2f}%",
    }
    st.dataframe(coef_df.style.format(fmt_dict), use_container_width=True)

with tab3:
    st.subheader("1. 데이터 전처리 5단계 파이프라인")
    st.markdown(
        """
- **1단계 (데이터 정제)**: 완전 중복 행 제거, 물리적 치수(`x, y, z`) 0값 및 극단 이상치(`y=58.9, 31.8 / z=31.8`) 제거로 회귀선 왜곡 차단
- **2단계 (다중공선성 제거)**: `x, y, z` 열 삭제로 `carat`과의 중복 상관성 제거 및 VIF 안정화(`1,112 → 1.15`)
- **3단계 (로그 변환)**: `carat`, `price` 자연로그 변환(`log_carat`, `log_price`)으로 비선형 분포 선형화 및 이분산성 완화
- **4단계 (범주형 인코딩)**: `cut`, `color`, `clarity` 원-핫 인코딩(`drop_first=True`, 17개 더미 생성)으로 기준 등급 대비 효과 정량화
- **5단계 (층화 분할)**: `log_price` 분위수 기준 80:20 층화 분할(`train.csv` / `test.csv`)로 고가 구간 분포 균형 유지
""")
