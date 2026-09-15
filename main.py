import datetime
import requests
import pandas as pd
import plotly.express as px
import pytz
import streamlit as st

# ==========================================
# 1. 페이지 기본 설정 및 제목
# ==========================================
st.set_page_config(
    page_title="박스오피스 종합 분석 대시보드",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 박스오피스 종합 분석 대시보드")
st.caption(
    "KOBIS API를 활용하여 일별 순위, 영역 차트, 다중 선그래프, 관객수 추이(이동평균 포함), 월별 관객수, 캘린더 히트맵을 분석합니다."
)


# ==========================================
# 2. 한국 시간 기준 '어제' 및 기간 기본값 설정
# ==========================================
def get_yesterday_date():
    """한국 시간(KST) 기준 어제 날짜(datetime.date)를 반환합니다."""
    kst = pytz.timezone("Asia/Seoul")
    now_kst = datetime.datetime.now(kst)
    yesterday = now_kst - datetime.timedelta(days=1)
    return yesterday.date()


yesterday_date = get_yesterday_date()

# 기본 기간을 최근 90일 전 ~ 어제로 설정
default_start_date = yesterday_date - datetime.timedelta(days=90)

# ==========================================
# 3. 사이드바: 날짜 및 기간 선택 (최대 선택 가능일: 어제)
# ==========================================
st.sidebar.header("🔍 데이터 조회 설정")

# [A] 수집 기간 선택
date_range = st.sidebar.date_input(
    label="1. 전체 데이터 수집 기간 (시작일, 종료일)",
    value=(default_start_date, yesterday_date),
    max_value=yesterday_date,
)

# 시작일과 종료일 지정 예외 처리
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
elif isinstance(date_range, tuple) and len(date_range) == 1:
    start_date = date_range[0]
    end_date = yesterday_date
else:
    start_date = date_range
    end_date = yesterday_date


# ==========================================
# 4. API 데이터 불러오기 함수 (캐싱 처리)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_single_day_box_office(api_key, target_date_str):
    """하루 치 박스오피스 데이터를 가져옵니다."""
    url = "https://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json"
    params = {"key": api_key, "targetDt": target_date_str}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "faultInfo" in data:
            return None, data["faultInfo"].get("message", "API 오류")

        daily_list = data.get("boxOfficeResult", {}).get(
            "dailyBoxOfficeList", []
        )
        return daily_list, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=3600)
def load_range_box_office_data(api_key, start_dt, end_dt):
    """지정한 전체 기간 동안의 데이터를 수집하여 하나의 DataFrame으로 통합합니다."""
    all_records = []
    current_dt = start_dt

    total_days = (end_dt - start_dt).days + 1
    if total_days <= 0:
        return pd.DataFrame(), "시작일이 종료일보다 늦을 수 없습니다."

    progress_bar = st.sidebar.progress(0)
    processed_days = 0

    while current_dt <= end_dt:
        dt_str = current_dt.strftime("%Y%m%d")
        daily_list, err = fetch_single_day_box_office(api_key, dt_str)

        if daily_list:
            for item in daily_list:
                item["targetDt"] = current_dt.strftime("%Y-%m-%d")
                all_records.append(item)

        processed_days += 1
        progress_bar.progress(processed_days / total_days)
        current_dt += datetime.timedelta(days=1)

    progress_bar.empty()

    if not all_records:
        return pd.DataFrame(), "그날은 아직 집계 전입니다."

    df = pd.DataFrame(all_records)

    # 숫자 데이터 형변환
    numeric_cols = [
        "rank",
        "rankInten",
        "audiCnt",
        "audiAcc",
        "scrnCnt",
        "showCnt",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df, None


# ==========================================
# 5. Secrets 인증키 검증 및 데이터 로드
# ==========================================
if "KOBIS_KEY" not in st.secrets:
    st.error("🔑 API 키를 찾을 수 없습니다.")
    st.info("Streamlit Secrets에 `KOBIS_KEY` 환경 변수를 설정해 주세요.")
    st.stop()

api_key = st.secrets["KOBIS_KEY"]

with st.spinner("선택하신 기간 동안의 데이터를 불러오는 중입니다..."):
    df_all, error_msg = load_range_box_office_data(api_key, start_date, end_date)

if error_msg or df_all.empty:
    if error_msg == "그날은 아직 집계 전입니다.":
        st.warning(f"⚠️ {error_msg}")
    else:
        st.error("⚠️ 데이터를 불러오지 못했습니다.")
        st.warning(f"**상세 원인:** {error_msg or '데이터가 존재하지 않습니다.'}")

        with st.expander("🛠️ 무엇을 확인해야 하나요?", expanded=True):
            st.markdown("""
            - **인증키(KOBIS_KEY) 확인**: Streamlit Secrets에 올바른 키가 등록되어 있는지 확인하세요.
            - **조회 날짜 확인**: 아직 영화관 집계가 완료되지 않은 날짜일 수 있습니다.
            """)
    st.stop()


# ==========================================
# 6. 데이터 전처리 및 선택 날짜 변경 기능
# ==========================================

# [기능] 순위 증감 화살표 가공
def format_rank_change(row):
    rank_old_and_new = row.get("rankOldAndNew", "OLD")
    inten = row.get("rankInten", 0)

    if rank_old_and_new == "NEW":
        return "🆕 NEW"
    elif inten > 0:
        return f"🔴 🔺 {inten}"
    elif inten < 0:
        return f"🔵 🔻 {abs(inten)}"
    else:
        return "-"


df_all["순위증감"] = df_all.apply(format_rank_change, axis=1)

# [기능] 100만 관객 돌파 🏆 이모지 추가
df_all["movieNm_display"] = df_all.apply(
    lambda r: f"{r['movieNm']} 🏆" if r["audiAcc"] >= 1_000_000 else r["movieNm"],
    axis=1,
)

# 1번, 2번에 보여줄 일별 특정 날짜 선택 (사이드바)
st.sidebar.markdown("---")
st.sidebar.subheader("📅 일별 박스오피스 상세 날짜 선택")

available_dates = sorted(df_all["targetDt"].unique(), reverse=True)

selected_dt_str = st.sidebar.selectbox(
    label="1, 2번 항목에서 확인할 날짜를 고르세요",
    options=available_dates,
    index=0,
)

selected_day_df = df_all[df_all["targetDt"] == selected_dt_str].sort_values("rank")


# ==========================================
# 7. 대시보드 구성 (1번 ~ 6번 전체)
# ==========================================

# ------------------------------------------
# 🥇 [1번] 선택 날짜의 1위 영화 지표 카드 세 장
# ------------------------------------------
if not selected_day_df.empty:
    top_1 = selected_day_df.iloc[0]

    st.markdown(f"### 🏆 1번: {selected_dt_str} 기준 박스오피스 1위")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="영화명", value=top_1["movieNm_display"])
    with col2:
        st.metric(label="당일 관객수", value=f"{top_1['audiCnt']:,} 명")
    with col3:
        st.metric(label="누적 관객수", value=f"{top_1['audiAcc']:,} 명")

    st.divider()

    # ------------------------------------------
    # ⛰️ [2번] 선택일 상위 영화 누적관객수 영역 차트
    # ------------------------------------------
    st.markdown(f"### ⛰️ 2번: {selected_dt_str} 기준 상위 영화의 누적관객수 변화 (영역 차트)")
    
    top_5_movies = selected_day_df.head(5)["movieNm"].tolist()
    area_df = df_all[df_all["movieNm"].isin(top_5_movies)].copy()
    area_df["dt"] = pd.to_datetime(area_df["targetDt"])
    area_df = area_df.sort_values("dt")

    fig2 = px.area(
        area_df,
        x="dt",
        y="audiAcc",
        color="movieNm_display",
        labels={
            "dt": "기준일자",
            "audiAcc": "누적관객수 (명)",
            "movieNm_display": "영화명",
        },
        title=f"선택일({selected_dt_str}) 상위 5개 영화의 누적관객수 성장 추이",
    )
    
    fig2.update_traces(hovertemplate="날짜: %{x|%Y-%m-%d}<br>누적관객수: %{y:,}명")
    fig2.update_layout(
        height=400,
        xaxis_title="기준일자",
        yaxis_title="누적관객수 (명)",
        legend_title="영화명",
    )
    
    st.plotly_chart(fig2, use_container_width=True)

    st.info("💡 **이 그래프로 알 수 있는 것**")
    with st.expander("📌 누적관객수 영역 차트 분석 가이드 보기", expanded=False):
        st.markdown("""
        1. **흥행 기울기(성장 속도):** 영역의 경사가 완만할 때는 관객 유입이 줄어든 상태이며, 경사가 가파를수록 흥행 속도가 빠른 구간입니다.
        2. **흥행 꺾임(입소문/롱런 여부):** 개봉 후 누적관객 그래프가 꺾이지 않고 꾸준히 대각선으로 상승하는지 파악하여 장기 흥행 여부를 분석할 수 있습니다.
        3. **상위 영화 간 성과 비교:** 선택한 날짜의 주요 상위권 영화들이 시간의 흐름에 따라 어떠한 격차로 누적 관객을 모았는지 상대적으로 비교 가능합니다.
        """)

    st.divider()

# ------------------------------------------
# 📈 [3번] 누적관객수 상위 5개 영화 다중 선그래프
# ------------------------------------------
st.markdown("### 📈 3번: 누적관객수 최고 상위 5개 영화의 일별 누적관객수 변화 (다중 선그래프)")

top_5_acc_movies = (
    df_all.groupby("movieNm")["audiAcc"].max().nlargest(5).index.tolist()
)

line_df = df_all[df_all["movieNm"].isin(top_5_acc_movies)].copy()
line_df["dt"] = pd.to_datetime(line_df["targetDt"])
line_df = line_df.sort_values("dt")

fig3 = px.line(
    line_df,
    x="dt",
    y="audiAcc",
    color="movieNm_display",
    markers=True,
    labels={
        "dt": "기준일자",
        "audiAcc": "누적관객수 (명)",
        "movieNm_display": "영화명",
    },
    title=f"전체 조회 기간 최고 누적관객수 TOP 5 영화 추이 ({start_date} ~ {end_date})",
)

fig3.update_traces(hovertemplate="날짜: %{x|%Y-%m-%d}<br>누적관객수: %{y:,}명")
fig3.update_layout(
    height=420,
    xaxis_title="기준일자",
    yaxis_title="누적관객수 (명)",
    legend_title="영화명",
)

st.plotly_chart(fig3, use_container_width=True)

st.info("💡 **이 그래프로 알 수 있는 것**")
with st.expander("📌 누적관객수 다중 선그래프 분석 가이드 보기", expanded=False):
    st.markdown("""
    1. **흥행 1위 다툼 및 역전 구간:** 상위 영화들 간의 누적 관객수 추세를 비교하여 어느 시점에 순위가 뒤바뀌었는지 파악할 수 있습니다.
    2. **관객 유입 지속성(롱런 여부):** 개봉 후 시간이 지나도 그래프 기울기가 지속적으로 완만하게라도 상승하면 장기 흥행에 성공했음을 의미합니다.
    3. **최종 흥행 규모 비교:** 수집된 전체 기간 동안 가장 높은 누적 관객을 모은 영화들의 최종 관객수 차이와 격차를 한눈에 볼 수 있습니다.
    """)

st.divider()

# ------------------------------------------
# 📈 [4번] 전체 기간 기준일자별 관객수 합계 추이 (7일 이동 평균선 포함)
# ------------------------------------------
st.markdown("### 📈 4번: 기준일자별 전체 관객수 합계 추이 및 7일 이동 평균")

daily_summary = (
    df_all.groupby("targetDt")["audiCnt"].sum().reset_index()
)
daily_summary.columns = ["기준일자", "일별전체관객수"]
daily_summary["dt"] = pd.to_datetime(daily_summary["기준일자"])

# 7일 이동 평균 계산 (최근 7일 간의 평균 관객수)
daily_summary["7일 이동 평균"] = (
    daily_summary["일별전체관객수"].rolling(window=7, min_periods=1).mean()
)

fig4 = px.line(
    daily_summary,
    x="dt",
    y=["일별전체관객수", "7일 이동 평균"],
    markers=True,
    labels={"dt": "날짜", "value": "관객수 (명)", "variable": "구분"},
    title=f"일별 전체 관객수 및 7일 이동 평균 추이 ({start_date} ~ {end_date})",
)

fig4.for_each_trace(
    lambda t: t.update(
        name="일별 관객수" if t.name == "일별전체관객수" else "7일 이동 평균"
    )
)
fig4.update_traces(hovertemplate="날짜: %{x|%Y-%m-%d}<br>관객수: %{y:,.0f}명")
fig4.update_layout(
    height=420,
    xaxis_title="기준일자",
    yaxis_title="관객수 (명)",
    legend_title="구분",
)

st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ------------------------------------------
# 📊 [5번] 월별(연-월) 전체 관객수 합계 막대그래프
# ------------------------------------------
st.markdown("### 📊 5번: 월별(연-월) 전체 관객수 합계")

daily_summary["연월"] = daily_summary["dt"].dt.strftime("%Y-%m")
monthly_summary = (
    daily_summary.groupby("연월")["일별전체관객수"].sum().reset_index()
)
monthly_summary.columns = ["연월", "월별관객수합계"]

fig5 = px.bar(
    monthly_summary,
    x="연월",
    y="월별관객수합계",
    text="월별관객수합계",
    labels={"연월": "조회 월", "월별관객수합계": "월총 관객수 (명)"},
    color="월별관객수합계",
    color_continuous_scale="Purples",
)
fig5.update_traces(texttemplate="%{text:,}명", textposition="outside")
fig5.update_layout(
    showlegend=False,
    height=400,
    coloraxis_showscale=False,
    xaxis_title="연-월 (Year-Month)",
    yaxis_title="총 관객수 (명)",
)

st.plotly_chart(fig5, use_container_width=True)

st.info("💡 **이 그래프로 알 수 있는 것**")
with st.expander("📌 월별 관객수 분석 가이드 보기", expanded=False):
    st.markdown("""
    1. **월별 영화 시장 규모 추이:** 특정 월에 극장을 찾은 총 방문객 수의 증감을 한눈에 파악할 수 있습니다.
    2. **시즌별 성수기 / 비수기 파악:** 여름 휴가철, 명절 연휴, 연말 등 성수기와 비수기의 총 관객수 차이를 비교할 수 있습니다.
    3. **대작 흥행 효과 검증:** 특정 월에 개봉한 대형 흥행작이 전체 영화 시장 규모를 얼마나 견인했는지 분석할 수 있습니다.
    """)

st.divider()

# ------------------------------------------
# 🗓️ [6번] 캘린더 히트맵 (월/주차별 x 요일별)
# ------------------------------------------
st.markdown("### 🗓️ 6번: 캘린더 히트맵 (월·주차별 × 요일별 관객수)")

days_ko = ["월", "화", "수", "목", "금", "토", "일"]
daily_summary["day_of_week_num"] = daily_summary["dt"].dt.dayofweek
daily_summary["요일"] = daily_summary["day_of_week_num"].map(lambda x: days_ko[x])

def get_month_week(dt):
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    return f"{dt.strftime('%Y-%m')} {(adjusted_dom - 1) // 7 + 1}주차"

daily_summary["월_주차"] = daily_summary["dt"].apply(get_month_week)
daily_summary["date_str"] = daily_summary["dt"].dt.strftime("%Y-%m-%d")

pivot_df = daily_summary.pivot(
    index="월_주차", columns="day_of_week_num", values="일별전체관객수"
)
pivot_dates = daily_summary.pivot(
    index="월_주차", columns="day_of_week_num", values="date_str"
)

pivot_df.columns = [days_ko[c] for c in pivot_df.columns]
pivot_dates.columns = [days_ko[c] for c in pivot_dates.columns]

col_order = [d for d in days_ko if d in pivot_df.columns]
pivot_df = pivot_df[col_order]
pivot_dates = pivot_dates[col_order]

hover_text = []
for i in range(len(pivot_df)):
    row_text = []
    for j in range(len(pivot_df.columns)):
        val = pivot_df.iloc[i, j]
        d_str = pivot_dates.iloc[i, j]
        if pd.isna(val) or pd.isna(d_str):
            row_text.append("데이터 없음")
        else:
            row_text.append(f"날짜: {d_str}<br>관객수: {int(val):,}명")
    hover_text.append(row_text)

fig6 = px.imshow(
    pivot_df,
    labels=dict(x="요일", y="월 - 주차", color="관객수"),
    x=pivot_df.columns,
    y=pivot_df.index,
    color_continuous_scale="Reds",
    aspect="auto",
)

fig6.update_traces(
    hoverongaps=False,
    hovertemplate="%{customdata}<extra></extra>",
    customdata=hover_text,
)

fig6.update_layout(
    height=max(350, len(pivot_df) * 35),
    xaxis_title="요일 (월 ~ 일)",
    yaxis_title="월 - 주차",
    coloraxis_colorbar=dict(title="관객수 (명)"),
)

st.plotly_chart(fig6, use_container_width=True)
