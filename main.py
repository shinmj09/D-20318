import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================
# 1. 페이지 기본 설정 및 제목
# ==========================================
st.set_page_config(
    page_title="영화별 박스오피스 관객수 분석", page_icon="🎬", layout="wide"
)

st.title("🎬 영화별 박스오피스 관객수 분석 대시보드")
st.caption(
    "1년간의 KOBIS 박스오피스 데이터를 기반으로 선택한 영화의 관객수 변화 추이를 시각화합니다."
)


# ==========================================
# 2. 데이터 불러오기 및 전처리 (캐싱 적용)
# ==========================================
# @st.cache_data: 한 번 불러온 데이터를 저장해두어 앱이 매번 새로 불러오는 것을 방지합니다.
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/keep-growing-park/data-science/refs/heads/main/dataset/kobis_1year_boxoffice.csv"

    # 1) CSV 파일 불러오기
    df = pd.read_csv(url)

    # 2) 결측치가 포함된 행 삭제
    df = df.dropna()

    # 3) "기준일자" 컬럼을 datetime 형식으로 변환
    df["기준일자"] = pd.to_datetime(df["기준일자"])

    # 4) 기준일자 오름차순 정렬
    df = df.sort_values(by="기준일자", ascending=True)

    return df


# 데이터 로드
with st.spinner("데이터를 불러오는 중입니다..."):
    df = load_and_preprocess_data()


# ==========================================
# 3. 영화 선택 기능 (사이드바)
# ==========================================
st.sidebar.header("🔍 검색 및 필터 설정")

# 영화 목록 추출: 각 영화의 '최고 누적관객수' 기준 내림차순 정렬
movie_rank = (
    df.groupby("영화명")["누적관객수"]
    .max()
    .sort_values(ascending=False)
    .index.tolist()
)

# 사용자 영화 선택 셀렉트박스 (기본값: 누적관객수가 가장 많은 1위 영화)
selected_movie = st.sidebar.selectbox(
    label="🎬 분석할 영화를 선택하세요 (누적관객수 순)",
    options=movie_rank,
    index=0,
)

# 선택한 영화의 데이터만 필터링
movie_df = df[df["영화명"] == selected_movie].sort_values(by="기준일자")


# ==========================================
# 4. 선택한 영화 기본 정보 요약 (지표 카드)
# ==========================================
if not movie_df.empty:
    st.subheader(f"📌 [{selected_movie}] 주요 정보")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="최고 누적관객수",
            value=f"{int(movie_df['누적관객수'].max()):,} 명",
        )
    with col2:
        st.metric(
            label="최고 일일관객수",
            value=f"{int(movie_df['해당일관객수'].max()):,} 명",
        )
    with col3:
        st.metric(
            label="집계 기간",
            value=f"{movie_df['기준일자'].dt.strftime('%Y-%m-%d').min()} ~ {movie_df['기준일자'].dt.strftime('%Y-%m-%d').max()}",
        )

st.divider()


# ==========================================
# 5. 구역 1: 선그래프 (일별 관객수 변화)
# ==========================================
st.markdown("### 📈 [구역 1] 일별 관객수 변화 추이 (선 그래프)")

if not movie_df.empty:
    # Plotly 선그래프 그리기
    fig1 = px.line(
        movie_df,
        x="기준일자",
        y="해당일관객수",
        markers=True,
        labels={"기준일자": "날짜", "해당일관객수": "해당일 관객수 (명)"},
        title=f"'{selected_movie}'의 기준일자별 해당일관객수 변화",
    )

    # 마우스 오버 툴팁 및 레이아웃 설정
    fig1.update_traces(hovertemplate="날짜: %{x|%Y-%m-%d}<br>관객수: %{y:,}명")
    fig1.update_layout(
        height=400,
        xaxis_title="기준일자",
        yaxis_title="관객수 (명)",
    )

    st.plotly_chart(fig1, use_container_width=True)

    # '이 그래프로 알 수 있는 것' 문구 영역
    max_date = movie_df.loc[
        movie_df["해당일관객수"].idxmax(), "기준일자"
    ].strftime("%Y-%m-%d")
    max_cnt = int(movie_df["해당일관객수"].max())

    st.info(
        f"💡 **이 그래프로 알 수 있는 것:** 영화 '{selected_movie}'은(는) **{max_date}**에 가장 많은 일일 관객수({max_cnt:,}명)를 기록하였으며, 날짜별 관객 유입의 피크 시점과 관객수 감소 추이를 파악할 수 있습니다."
    )
else:
    st.warning("선택한 영화의 데이터가 존재하지 않습니다.")

st.divider()


# ==========================================
# 6. 구역 2: 영역차트 (누적관객수 변화)
# ==========================================
st.markdown("### ⛰️ [구역 2] 누적관객수 변화 추이 (영역 차트)")

if not movie_df.empty:
    # Plotly 영역차트(area chart) 그리기
    fig2 = px.area(
        movie_df,
        x="기준일자",
        y="누적관객수",
        labels={"기준일자": "날짜", "누적관객수": "누적 관객수 (명)"},
        title=f"'{selected_movie}'의 기준일자별 누적관객수 성장 추이",
    )

    # 마우스 오버 툴팁 및 레이아웃 설정
    fig2.update_traces(
        hovertemplate="날짜: %{x|%Y-%m-%d}<br>누적관객수: %{y:,}명",
        fillcolor="rgba(31, 119, 180, 0.4)",  # 영역 반투명 색상 설정
    )
    fig2.update_layout(
        height=400,
        xaxis_title="기준일자",
        yaxis_title="누적 관객수 (명)",
    )

    st.plotly_chart(fig2, use_container_width=True)

    # '이 그래프로 알 수 있는 것' 문구 영역
    final_acc = int(movie_df["누적관객수"].max())

    st.info(
        f"💡 **이 그래프로 알 수 있는 것:** 영화 '{selected_movie}'은(는) 최종적으로 **{final_acc:,}명**의 누적 관객을 동원했습니다. 영역의 경사가 가파를수록 관객 모집 속도가 빨랐음을 의미하며, 평평해지는 구간을 통해 흥행 모멘텀이 완화된 시점을 분석할 수 있습니다."
    )
else:
    st.warning("선택한 영화의 데이터가 존재하지 않습니다.")

st.divider()


# ==========================================
# 7. 구역 3: 다중 선그래프 (20일 이상 TOP10 등재 + 누적관객수 TOP 5 영화 비교)
# ==========================================
st.markdown("### 📊 [구역 3] 롱런 흥행작(20일 이상 TOP10) 누적관객수 TOP 5 비교")

# 1) 영화별 TOP10 차트 등재 일수(행 개수) 집계
movie_appear_counts = df.groupby("영화명")["기준일자"].count()

# 2) 등재 일수가 20일 이상인 영화 목록만 필터링
long_run_movies = movie_appear_counts[movie_appear_counts >= 20].index

# 3) 20일 이상 등재된 영화 중 누적관객수가 가장 높은 상위 5개 영화 선택
top5_long_run_movies = (
    df[df["영화명"].isin(long_run_movies)]
    .groupby("영화명")["누적관객수"]
    .max()
    .nlargest(5)
    .index.tolist()
)

# 4) 해당 상위 5개 영화 데이터 추출 및 정렬
top5_df = df[df["영화명"].isin(top5_long_run_movies)].sort_values(by="기준일자")

if not top5_df.empty:
    # Plotly 다중 선그래프 그리기
    fig3 = px.line(
        top5_df,
        x="기준일자",
        y="누적관객수",
        color="영화명",
        markers=True,
        labels={"기준일자": "날짜", "누적관객수": "누적 관객수 (명)", "영화명": "영화 제목"},
        title="20일 이상 TOP 10 유지 영화 중 최고 누적관객수 TOP 5 비교",
    )

    # 마우스 오버 툴팁 및 레이아웃 설정
    fig3.update_traces(
        hovertemplate="영화: %{fullData.name}<br>날짜: %{x|%Y-%m-%d}<br>누적관객수: %{y:,}명"
    )
    fig3.update_layout(
        height=450,
        xaxis_title="기준일자",
        yaxis_title="누적 관객수 (명)",
        legend_title="TOP 5 롱런 영화",
    )

    st.plotly_chart(fig3, use_container_width=True)

    # TOP 5 영화 이름 가공
    top5_str = ", ".join([f"**{m}**" for m in top5_long_run_movies])

    st.info(
        f"💡 **이 그래프로 알 수 있는 것:** 박스오피스 TOP 10에 20일 이상 지속 진입한 대표 롱런 영화 중 관객 동원력이 높은 상위 5개 영화({top5_str})의 누적관객수 비교입니다. 단순 단기 흥행에 그치지 않고 꾸준히 관객을 모은 대표 대작들의 흥행 곡선과 기울기 차이를 분석할 수 있습니다."
    )
else:
    st.warning("조건에 맞는 TOP 5 영화 데이터를 불러올 수 없습니다.")

st.divider()


# ==========================================
# 8. 구역 4: 이동평균선 (전체 TOP10 일별 관객수 합계 & 7일 이동평균)
# ==========================================
st.markdown("### 📈 [구역 4] 전체 박스오피스 일별 관객수 추이 및 7일 이동평균선")

# 1) 기준일자별 TOP10 영화 전체의 해당일관객수 합계 계산
daily_total = (
    df.groupby("기준일자")["해당일관객수"].sum().reset_index()
)

# 2) 7일 이동평균 계산 (rolling window=7)
daily_total["7일 이동평균"] = (
    daily_total["해당일관객수"].rolling(window=7, min_periods=1).mean()
)

if not daily_total.empty:
    # 3) Plotly 선그래프 생성
    fig4 = px.line(
        daily_total,
        x="기준일자",
        y=["해당일관객수", "7일 이동평균"],
        labels={"기준일자": "날짜", "value": "관객수 (명)", "variable": "구분"},
        title="전체 박스오피스 일별 총관객수 및 7일 이동평균 추이",
    )

    # 4) 선 스타일 지정
    fig4.data[0].update(
        name="일별 총관객수 (원본)",
        line=dict(color="rgba(180, 180, 180, 0.4)", width=1.5),
        hovertemplate="날짜: %{x|%Y-%m-%d}<br>일별 총관객수: %{y:,}명",
    )
    fig4.data[1].update(
        name="7일 이동평균",
        line=dict(color="#E63946", width=3),
        hovertemplate="날짜: %{x|%Y-%m-%d}<br>7일 이동평균: %{y:,.0f}명",
    )

    fig4.update_layout(
        height=450,
        xaxis_title="기준일자",
        yaxis_title="관객수 (명)",
        legend_title="구분",
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.info(
        "💡 **이 그래프로 알 수 있는 것:** 주말과 평일 간 관객수 변동성이 큰 원본 데이터(연한 회색선)의 노이즈를 제하고, 7일 이동평균선(진한 빨간선)을 통해 한국 영화 시장 전체의 중장기적인 관객 유입 흐름과 성수기·비수기 추세를 명확히 파악할 수 있습니다."
    )
else:
    st.warning("전체 일별 관객수 데이터를 불러올 수 없습니다.")

st.divider()


# ==========================================
# 9. 구역 5: 월별 막대그래프 (월별 박스오피스 전체 관객수 합산)
# ==========================================
st.markdown("### 📊 [구역 5] 월별 박스오피스 전체 관객수 합계 (막대그래프)")

if not daily_total.empty:
    # 1) 기준일자를 기준으로 '연-월(YYYY-MM)' 컬럼 생성 및 월별 합산
    monthly_total = daily_total.copy()
    monthly_total["연월"] = monthly_total["기준일자"].dt.strftime("%Y-%m")

    monthly_df = (
        monthly_total.groupby("연월")["해당일관객수"]
        .sum()
        .reset_index()
        .sort_values(by="연월")
    )

    # 2) Plotly 막대그래프 그리기
    fig5 = px.bar(
        monthly_df,
        x="연월",
        y="해당일관객수",
        text="해당일관객수",
        color="해당일관객수",
        color_continuous_scale="Blues",
        labels={"연월": "기준 월", "해당일관객수": "월간 총 관객수 (명)"},
        title="월별 박스오피스 총 관객수 집계",
    )

    # 3) 막대 위 텍스트 세 자릿수 콤마 설정 및 레이아웃 정리
    fig5.update_traces(
        texttemplate="%{text:,}명",
        textposition="outside",
        hovertemplate="월: %{x}<br>총 관객수: %{y:,}명",
    )
    fig5.update_layout(
        height=450,
        xaxis_title="연-월",
        yaxis_title="월간 총 관객수 (명)",
        coloraxis_showscale=False,
    )

    st.plotly_chart(fig5, use_container_width=True)

    # 최고 관객수 기록 월 계산
    best_month_row = monthly_df.loc[monthly_df["해당일관객수"].idxmax()]
    best_month = best_month_row["연월"]
    best_month_cnt = int(best_month_row["해당일관객수"])

    st.info(
        f"💡 **이 그래프로 알 수 있는 것:** 1년간의 박스오피스 월별 총 관객 수 집계입니다. 가장 많은 관객이 극장을 찾은 달은 **{best_month}**({best_month_cnt:,}명)이며, 월별 비교를 통해 연중 극장가의 주요 성수기와 비수기 시즌을 한눈에 식별할 수 있습니다."
    )
else:
    st.warning("월별 관객수 데이터를 생성할 수 없습니다.")

st.divider()


# ==========================================
# 10. [추가] 구역 6: 캘린더 히트맵 (주차별 x 요일별 관객수 분포)
# ==========================================
st.markdown("### 🗓️ [구역 6] 일별 관객수 분포 (캘린더 히트맵)")

if not daily_total.empty:
    cal_df = daily_total.copy()

    # 1) 요일 및 주차 파생 변수 생성
    # dt.weekday: 월요일=0, 화요일=1, ..., 일요일=6
    day_names = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    cal_df["요일"] = cal_df["기준일자"].dt.weekday.map(lambda x: day_names[x])

    # ISO 주차(YYYY-Www) 추출
    cal_df["주차"] = cal_df["기준일자"].dt.strftime("%Y-%V주차")

    # 툴팁에 사용할 yyyy-mm-dd 날짜 텍스트 컬럼
    cal_df["날짜텍스트"] = cal_df["기준일자"].dt.strftime("%Y-%m-%d")

    # 2) Plotly 히트맵 생성 (x: 주차, y: 요일)
    fig6 = px.density_heatmap(
        cal_df,
        x="주차",
        y="요일",
        z="해당일관객수",
        category_orders={"요일": day_names},  # 요일 순서를 월~일로 지정
        color_continuous_scale="Reds",
        hover_data={"날짜텍스트": True, "주차": False, "요일": False},
        labels={"주차": "연도별 주차", "요일": "요일", "해당일관객수": "일관객수"},
        title="주차별 × 요일별 일일 총 관객수 히트맵",
    )

    # 3) 마우스 커서 호버 툴팁 포맷 설정 (yyyy-mm-dd 표출)
    fig6.update_traces(
        hovertemplate="<b>날짜: %{customdata[0]}</b><br>요일: %{y}<br>관객수: %{z:,}명<extra></extra>"
    )

    fig6.update_layout(
        height=400,
        xaxis_title="주차",
        yaxis_title="요일",
        coloraxis_colorbar=dict(title="관객수 (명)"),
    )

    st.plotly_chart(fig6, use_container_width=True)

    st.info(
        "💡 **이 그래프로 알 수 있는 것:** 날짜별 극장 관객 동원력을 달력 격자 형태의 색상 농도로 시각화했습니다. 붉은색이 짙을수록 관객수가 많았던 날(주말, 연휴, 특정 신작 개봉일 등)을 의미하며, 마우스를 올리면 정확한 날짜(`YYYY-MM-DD`)와 관객수를 확인할 수 있습니다."
    )
else:
    st.warning("캘린더 히트맵 데이터를 생성할 수 없습니다.")
