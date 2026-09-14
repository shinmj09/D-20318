import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 페이지 기본 설정
st.set_page_config(
    page_title="박스오피스 데이터 대시보드",
    page_icon="🎬",
    layout="wide"
)

# -------------------------------------------------------------
# [1. 데이터 불러오기 및 2. 날짜 전처리]
# @st.cache_data를 사용해 데이터를 한 번만 읽어오고 메모리에 캐싱(저장)합니다.
# -------------------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    csv_url = "https://raw.githubusercontent.com/keep-growing-park/data-science/refs/heads/main/dataset/kobis_1year_boxoffice.csv"
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_url)
    
    # 결측치(빈 값)가 포함된 행 삭제
    df = df.dropna()
    
    # '기준일자' 컬럼을 datetime(날짜) 형태로 변환
    df["기준일자"] = pd.to_datetime(df["기준일자"])
    
    # 기준일자 오름차순으로 데이터 정렬
    df = df.sort_values(by="기준일자").reset_index(drop=True)
    
    return df

# 데이터 로드
df = load_and_preprocess_data()

# -------------------------------------------------------------
# [3. 영화 선택 기능 (사이드바)]
# 영화별 최대 누적관객수를 기준으로 정렬하여 선택 목록을 만듭니다.
# -------------------------------------------------------------
st.sidebar.header("🔍 필터 설정")

# 영화별 최고 '누적관객수'를 기준으로 내림차순 정렬된 목록 생성
movie_max_audience = (
    df.groupby("영화명")["누적관객수"]
    .max()
    .sort_values(ascending=False)
)
movie_list = movie_max_audience.index.tolist()

# 사이드바에서 영화 선택
selected_movie = st.sidebar.selectbox(
    "조회할 영화를 선택하세요:",
    options=movie_list,
    index=0
)

# 선택한 영화의 데이터만 필터링
filtered_df = df[df["영화명"] == selected_movie]

# -------------------------------------------------------------
# 메인 대시보드 화면
# -------------------------------------------------------------
st.title("🎬 KOBIS 박스오피스 분석 대시보드")
st.write(f"현재 선택된 영화: **{selected_movie}**")

# 상단 요약 정보 (최고 누적관객수, 상영 데이터 기간 등)
col_m1, col_m2 = st.columns(2)
with col_m1:
    max_audience = filtered_df["누적관객수"].max()
    st.metric(label="최종 누적관객수", value=f"{max_audience:,}명")
with col_m2:
    date_range = f"{filtered_df['기준일자'].min().strftime('%Y-%m-%d')} ~ {filtered_df['기준일자'].max().strftime('%Y-%m-%d')}"
    st.metric(label="집계 기간", value=date_range)

st.divider()

# -------------------------------------------------------------
# [섹션 1] 일별 관객수 변화 (선 그래프)
# -------------------------------------------------------------
st.subheader("1. 일별 관객수 추이")

# Plotly 단일 선 그래프 생성
fig_line = px.line(
    filtered_df,
    x="기준일자",
    y="해당일관객수",
    title=f"'{selected_movie}' 일별 관객수 추이",
    markers=True,
    labels={"기준일자": "날짜", "해당일관객수": "관객수(명)"}
)

fig_line.update_layout(hovermode="x unified")
st.plotly_chart(fig_line, use_container_width=True)

st.info(
    f"💡 **이 그래프로 알 수 있는 것:** "
    f"개봉 이후 요일별(주말 스파이크 등) 관객 유입 패턴과 흥행 화력의 감소 추세를 한눈에 파악할 수 있습니다."
)

st.divider()

# -------------------------------------------------------------
# [섹션 2] 누적 관객수 추이 (영역 차트)
# -------------------------------------------------------------
st.subheader("2. 누적 관객수 추이 (영역 차트)")

fig_area = px.area(
    filtered_df,
    x="기준일자",
    y="누적관객수",
    title=f"'{selected_movie}' 누적 관객수 증가 추이",
    labels={"기준일자": "날짜", "누적관객수": "누적 관객수(명)"}
)

fig_area.update_layout(hovermode="x unified")
st.plotly_chart(fig_area, use_container_width=True)

st.info(
    f"💡 **이 그래프로 알 수 있는 것:** "
    f"시간 경과에 따라 전체 관객수가 누적되는 증가율(완만해지는 시점)과 최종 도달 규모를 시각적으로 직관적이게 확인할 수 있습니다."
)

st.divider()

# -------------------------------------------------------------
# [섹션 3] 장기 흥행 TOP 5 영화 누적 관객수 비교 (다중 선 그래프)
# -------------------------------------------------------------
st.subheader("3. 20일 이상 차트인 영화 중 흥행 TOP 5 누적 관객수 비교")

chart_in_days = df["영화명"].value_counts()
movies_over_20days = chart_in_days[chart_in_days >= 20].index

top5_long_run_movies = (
    df[df["영화명"].isin(movies_over_20days)]
    .groupby("영화명")["누적관객수"]
    .max()
    .sort_values(ascending=False)
    .head(5)
    .index.tolist()
)

top5_long_run_df = df[df["영화명"].isin(top5_long_run_movies)]

fig_multi_line = px.line(
    top5_long_run_df,
    x="기준일자",
    y="누적관객수",
    color="영화명",
    title="20일 이상 차트인 영화 중 누적 관객수 상위 5편의 성장 추이",
    labels={"기준일자": "날짜", "누적관객수": "누적 관객수(명)", "영화명": "영화 제목"},
    markers=False
)

fig_multi_line.update_layout(
    hovermode="x unified",
    legend=dict(
        title="영화 목록",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig_multi_line, use_container_width=True)

st.info(
    "💡 **이 그래프로 알 수 있는 것:** "
    "단기 반짝 흥행작(20일 미만)을 제외하고, 최소 20일 이상 차트 순위를 유지하며 "
    "안정적인 롱런(Long-run) 흥행을 이어간 대표작 5편의 누적 스코어 증가 속도와 최종 규모를 객관적으로 비교할 수 있습니다."
)

st.divider()

# -------------------------------------------------------------
# [섹션 4] 전체 박스오피스 총 관객수 및 7일 이동평균선
# -------------------------------------------------------------
st.subheader("4. 전체 박스오피스 일일 총관객수 및 7일 이동평균선")

# 1) 기준일자별 TOP 10 영화의 일일 총관객수 합계 계산
daily_total_df = (
    df.groupby("기준일자")["해당일관객수"]
    .sum()
    .reset_index()
    .rename(columns={"해당일관객수": "일일총관객수"})
)

# 2) 7일 이동평균 계산 (min_periods=1로 초반 결측치 방지)
daily_total_df["7일이동평균"] = daily_total_df["일일총관객수"].rolling(window=7, min_periods=1).mean()

# 3) Plotly graph_objects로 원본 선(연하게)과 이동평균선(진하게) 겹쳐 그리기
fig_ma = go.Figure()

# 원본 일일 총관객수 (연한 회색/하늘색 계열, 얇은 선)
fig_ma.add_trace(
    go.Scatter(
        x=daily_total_df["기준일자"],
        y=daily_total_df["일일총관객수"],
        mode="lines",
        name="일일 총관객수 (원본)",
        line=dict(color="rgba(150, 165, 180, 0.4)", width=1.5),
        hovertemplate="날짜: %{x|%Y-%m-%d}<br>일일 총관객수: %{y:,.0f}명<extra></extra>"
    )
)

# 7일 이동평균선 (진한 파란색 계열, 굵은 선)
fig_ma.add_trace(
    go.Scatter(
        x=daily_total_df["기준일자"],
        y=daily_total_df["7일이동평균"],
        mode="lines",
        name="7일 이동평균",
        line=dict(color="#1f77b4", width=3),
        hovertemplate="날짜: %{x|%Y-%m-%d}<br>7일 이동평균: %{y:,.0f}명<extra></extra>"
    )
)

# 레이아웃 설정
fig_ma.update_layout(
    title="기준일자별 박스오피스 총관객수 및 7일 이동평균 추세",
    xaxis_title="날짜",
    yaxis_title="관객수(명)",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# 스트림릿에 차트 출력
st.plotly_chart(fig_ma, use_container_width=True)

# '이 그래프로 알 수 있는 것' 문구 영역
st.info(
    "💡 **이 그래프로 알 수 있는 것:** "
    "주말마다 치솟고 평일에 급감하는 요일별 노이즈(진폭)를 7일 이동평균선으로 부드럽게 정제하여, "
    "극장가 전체의 성수기·비수기 사이클과 장기적인 영화 관람 수요 흐름을 뚜렷하게 확인할 수 있습니다."
)
