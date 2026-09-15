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
    page_title="일별 박스오피스 조회", page_icon="🎬", layout="wide"
)

st.title("🎬 일별 박스오피스 조회")
st.caption(
    "KOBIS API를 활용하여 원하는 날짜의 박스오피스 정보를 보여줍니다."
)


# ==========================================
# 2. 한국 시간 기준 '어제' 날짜 구하기
# ==========================================
def get_yesterday_date():
    """배포 서버 시계와 무관하게 한국 시간(KST) 기준 어제 날짜(datetime.date)를 반환합니다."""
    kst = pytz.timezone("Asia/Seoul")
    now_kst = datetime.datetime.now(kst)
    yesterday = now_kst - datetime.timedelta(days=1)
    return yesterday.date()


yesterday_date = get_yesterday_date()

# ==========================================
# 3. 날짜 선택 달력 (최대 선택 가능일: 어제)
# ==========================================
selected_date = st.date_input(
    label="📅 조회할 날짜를 선택하세요 (오늘 날짜는 집계 전입니다)",
    value=yesterday_date,
    max_value=yesterday_date,  # 오늘 이후 날짜 선택 불가
)

# API 요청용 여덟 자리 문자열 변환 (YYYYMMDD)
target_date_str = selected_date.strftime("%Y%m%d")


# ==========================================
# 4. API 데이터 불러오기 함수 (캐싱 처리)
# ==========================================
# 날짜별로 1시간(3600초) 동안 결과를 저장하여 동일 날짜 재요청 시 API 호출을 방지합니다.
@st.cache_data(ttl=3600)
def fetch_box_office_data(api_key, target_date):
    url = "https://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json"
    params = {"key": api_key, "targetDt": target_date}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # 1) faultInfo 예외 상자가 있는 경우
        if "faultInfo" in data:
            error_message = data["faultInfo"].get(
                "message", "알 수 없는 API 오류"
            )
            return None, f"API 오류가 발생했습니다: {error_message}"

        # 2) 영화 목록 가져오기
        box_office_result = data.get("boxOfficeResult", {})
        daily_list = box_office_result.get("dailyBoxOfficeList", [])

        # 목록이 비어 있는 경우
        if not daily_list:
            return None, "그날은 아직 집계 전입니다."

        return daily_list, None

    except requests.exceptions.RequestException as e:
        return None, f"네트워크 요청 실패: {str(e)}"


# ==========================================
# 5. Secrets 인증키 검증
# ==========================================
if "KOBIS_KEY" not in st.secrets:
    st.error("🔑 API 키를 찾을 수 없습니다.")
    st.info(
        "Streamlit Cloud의 Secrets 설정에서 `KOBIS_KEY` 환경 변수를 등록해 주세요."
    )
    st.stop()

api_key = st.secrets["KOBIS_KEY"]

# 데이터 로드
daily_list, error_msg = fetch_box_office_data(api_key, target_date_str)

# ==========================================
# 6. 에러 및 비어있는 데이터 처리
# ==========================================
if error_msg:
    if error_msg == "그날은 아직 집계 전입니다.":
        st.warning(f"⚠️ {error_msg}")
    else:
        st.error("⚠️ 데이터를 불러오지 못했습니다.")
        st.warning(f"**상세 원인:** {error_msg}")

        with st.expander("🛠️ 무엇을 확인해야 하나요?", expanded=True):
            st.markdown("""
            - **발급받은 인증키(KOBIS_KEY)가 정확한지 확인해 주세요.**
            - **KOBIS API 서비스 상태를 확인해 주세요.**
            """)
    st.stop()

# ==========================================
# 7. 데이터 전처리 (형변환 및 기호 가공)
# ==========================================
df = pd.DataFrame(daily_list)

# 문자열 숫자를 정수형으로 변환
numeric_columns = [
    "rank",
    "rankInten",
    "audiCnt",
    "audiAcc",
    "scrnCnt",
    "showCnt",
]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

# 순위 기준 정렬
df = df.sort_values(by="rank", ascending=True)


# [기능 1] 순위 증감(rankInten) 화살표 텍스트 가공 함수
def format_rank_change(row):
    rank_old_and_new = row.get("rankOldAndNew", "OLD")
    inten = row.get("rankInten", 0)

    if rank_old_and_new == "NEW":
        return "🆕 NEW"
    elif inten > 0:
        return f"🔴 🔺 {inten}"  # 순위 상승 (빨간 위 화살표)
    elif inten < 0:
        return f"🔵 🔻 {abs(inten)}"  # 순위 하강 (파란 아래 화살표)
    else:
        return "-"


df["순위증감"] = df.apply(format_rank_change, axis=1)


# [기능 2] 누적관객 100만 명 이상 트로피(🏆) 이모지 추가 함수
def format_movie_name(row):
    movie_name = row["movieNm"]
    acc_audiences = row["audiAcc"]
    if acc_audiences >= 1_000_000:
        return f"{movie_name} 🏆"
    return movie_name


df["movieNm_display"] = df.apply(format_movie_name, axis=1)

# ==========================================
# 8. 대시보드 화면 구성
# ==========================================

# --- [A] 1위 영화 지표 카드 ---
top_1 = df.iloc[0]

st.markdown(f"### 🏆 {selected_date.strftime('%Y-%m-%d')} 박스오피스 1위")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="영화명", value=top_1["movieNm_display"])
with col2:
    st.metric(label="당일 관객수", value=f"{top_1['audiCnt']:,} 명")
with col3:
    st.metric(label="누적 관객수", value=f"{top_1['audiAcc']:,} 명")

st.divider()

# --- [B] 상위 5개 영화 관객수 막대그래프 ---
st.markdown("### 📊 관객수 상위 5개 영화")
top_5_df = df.head(5).sort_values(by="rank", ascending=False)

fig = px.bar(
    top_5_df,
    x="audiCnt",
    y="movieNm_display",
    orientation="h",
    text="audiCnt",
    labels={"audiCnt": "당일 관객수 (명)", "movieNm_display": "영화명"},
    color="audiCnt",
    color_continuous_scale="Blues",
)

fig.update_traces(texttemplate="%{text:,}명", textposition="outside")
fig.update_layout(
    showlegend=False,
    height=350,
    xaxis_title="관객수 (명)",
    yaxis_title="",
    coloraxis_showscale=False,
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- [C] 전체 박스오피스 순위 표 ---
st.markdown("### 📋 박스오피스 전체 순위")

# 표출용 컬럼 정리
display_df = df[
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
