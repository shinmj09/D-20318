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
    page_title="일별/월별 박스오피스 분석 대시보드",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 박스오피스 데이터 분석 대시보드")
st.caption(
    "KOBIS API를 활용하여 기간별 박스오피스 추이 및 월별 관객수를 분석합니다."
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

# 기본 기간 설정: 90일 전 ~ 어제 (월별 추이 확인을 위해 여러 달 데이터 선택 가능하도록 설정)
default_start_date = yesterday_date - datetime.timedelta(days=90)

# ==========================================
# 3. 날짜 범위 선택 달력 (최대 선택 가능일: 어제)
# ==========================================
st.sidebar.header("🔍 조회 기간 설정")
date_range = st.sidebar.date_input(
    label="조회할 기간을 선택하세요 (시작일, 종료일)",
    value=(default_start_date, yesterday_date),
    max_value=yesterday_date,
)

# 시작일과 종료일 지정 예외 처리
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = date_range[0]
    end_date = yesterday_date


# ==========================================
# 4. API 데이터 불러오기 함수 (캐싱 처리)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_single_day_box_office(api_key, target_date_str):
    """하루 치 박스오피스 데이터를 가져오는 함수"""
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
    """선택한 전체 기간 동안의 데이터를 수집하여 하나의 DataFrame으로 통합합니다."""
    all_records = []
    current_dt = start_dt

    # 날짜 범위 수집 진행 상황 표시용 Progress bar
    progress_bar = st.sidebar.progress(0)
    total_days = (end_dt - start_dt).days + 1
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
        return pd.DataFrame()

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
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ==========================================
# 5. Secrets 인증키 검증 및 데이터 수집
# ==========================================
if "KOBIS_KEY" not in st.secrets:
    st.error("🔑 API 키를 찾을 수 없습니다.")
    st.info("Streamlit Secrets에 `KOBIS_KEY` 환경 변수를 설정해 주세요.")
    st.stop()

api_key = st.secrets["KOBIS_KEY"]

with st.spinner("선택하신 기간 동안의 데이터를 불러오는 중입니다..."):
    df_all = load_range_box_office_data(api_key, start_date, end_date)

if df_all.empty:
    st.warning("⚠️ 선택하신 기간의 데이터가 비어 있거나 아직 집계 전입니다.")
    st.stop()


# ==========================================
# 6. 전처리 및 시각화 데이터 준비
# ==========================================

# 1) 순위 증감 및 표시용 영화명 가공
def format_rank_change(row):
    rank_old_and_new = row.get("rankOldAndNew", "OLD")
    inten = row.get("rankInten", 0)
    if rank_old_and_new == "NEW":
        return "🆕 NEW"
    elif inten > 0:
        return f"🔴 🔺 {int(inten)}"
    elif inten < 0:
        return f"🔵 🔻 {int(abs(inten))}"
    else:
        return "-"


df_all["순위증감"] = df_all.apply(format_rank_change, axis=1)
df_all["movieNm_display"] = df_all.apply(
    lambda r: f"{r['movieNm']} 🏆" if r["audiAcc"] >= 1_000_000 else r["movieNm"],
    axis=1,
)

# 2) 네 번째 그래프용: 기준일자별 전체 관객수 합계
daily_summary = (
    df_all.groupby("targetDt")["audiCnt"].sum().reset_index()
)
daily_summary.columns = ["기준일자", "일별전체관객수"]
daily_summary["기준일자"] = pd.to_datetime(daily_summary["기준일자"])

# 3) 다섯 번째 그래프용: 월(연-월) 단위 재그룹화 및 합산
daily_summary["연월"] = daily_summary["기준일자"].dt.strftime("%Y-%m")
monthly_summary = (
    daily_summary.groupby("연월")["일별전체관객수"].sum().reset_index()
)
monthly_summary.columns = ["연월", "월별관객수합계"]


# ==========================================
# 7. 대시보드 화면 구성
# ==========================================

# --- 최근 선택일(종료일) 기준 1위 및 상위 5개 표출 ---
latest_dt_str = end_date.strftime("%Y-%m-%d")
latest_df = df_all[df_all["targetDt"] == latest_dt_str].sort_values("rank")

if not latest_df.empty:
    top_1 = latest_df.iloc[0]

    st.markdown(f"### 🏆 {latest_dt_str} 기준 박스오피스 1위")
    c1, c2, c3 = st.columns(3)
    c1.metric(label="영화명", value=top_1["movieNm_display"])
    c2.metric(label="당일 관객수", value=f"{int(top_1['audiCnt']):,} 명")
    c3.metric(label="누적 관객수", value=f"{int(top_1['audiAcc']):,} 명")

    st.divider()

    st.markdown(f"### 📊 {latest_dt_str} 기준 관객수 상위 5개 영화")
    top_5_df = latest_df.head(5).sort_values(by="rank", ascending=False)

    fig1 = px.bar(
        top_5_df,
        x="audiCnt",
        y="movieNm_display",
        orientation="h",
        text="audiCnt",
        labels={"audiCnt": "당일 관객수 (명)", "movieNm_display": "영화명"},
        color="audiCnt",
        color_continuous_scale="Blues",
    )
    fig1.update_traces(texttemplate="%{text:,}명", textposition="outside")
    fig1.update_layout(
        showlegend=False, height=300, coloraxis_showscale=False
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

# --- 네 번째 그래프: 일별 관객수 추이 ---
st.markdown("### 📈 네 번째 그래프: 기준일자별 전체 관객수 합계 추이")
fig4 = px.line(
    daily_summary,
    x="기준일자",
    y="일별전체관객수",
    markers=True,
    labels={"기준일자": "날짜", "일별전체관객수": "전체 관객수 (명)"},
    title=f"일별 전체 관객수 변화 ({start_date} ~ {end_date})",
)
fig4.update_traces(hovertemplate="날짜: %{x|%Y-%m-%d}<br>관객수: %{y:,}명")
st.plotly_chart(fig4, use_container_width=True)

st.divider()

# --- [요청 사항] 다섯 번째 그래프: 월별 관객수 합계 막대그래프 ---
st.markdown("### 📊 다섯 번째 그래프: 월별(연-월) 전체 관객수 합계")

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

# --- 그래프 해석 및 가이드 영역 ---
st.info("💡 **이 그래프로 알 수 있는 것**")
with st.expander("📌 월별 관객수 분석 가이드 보기", expanded=True):
    st.markdown("""
    1. **월별 영화 시장 규모 추이:**
       - 특정 월에 총 극장 방문객 수가 급증했는지 혹은 감소했는지 한눈에 파악할 수 있습니다.
    2. **시즌별 성수기 / 비수기 파악:**
       - 여름 휴가철, 명절 연휴, 연말 등 영화계 주요 성수기 월과 비수기 월의 총 관객수 차이를 비교해 볼 수 있습니다.
    3. **대작 흥행 효과 검증:**
       - 해당 월에 개봉한 대형 흥행작(천만 관객 영화 등)이 전체 영화 시장의 규모를 얼마나 견인했는지 직관적으로 분석할 수 있습니다.
    """)

st.divider()

# --- 박스오피스 표출 ---
if not latest_df.empty:
    st.markdown(f"### 📋 {latest_dt_str} 전체 박스오피스 순위")
    display_df = latest_df[
        [
            "rank",
            "순위증감",
            "movieNm_display",
            "openDt",
            "audiCnt",
            "audiAcc",
            "scrnCnt",
        ]
    ].copy()
    display_df.columns = [
        "순위",
        "전날 대비",
        "영화명",
        "개봉일",
        "당일 관객수",
        "누적 관객수",
        "스크린수",
    ]

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "순위": st.column_config.NumberColumn(format="%d위"),
            "당일 관객수": st.column_config.NumberColumn(format="%d명"),
            "누적 관객수": st.column_config.NumberColumn(format="%d명"),
            "스크린수": st.column_config.NumberColumn(format="%d개"),
        },
    )
