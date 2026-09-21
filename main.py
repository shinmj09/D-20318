import streamlit as st
import pandas as pd
import plotly.express as px

# 페이지 설정
st.set_page_config(page_title="영화 데이터 그래프 도감 2 - 분포와 관계", layout="wide")

# 제목
st.title("영화 데이터 그래프 도감 2 - 분포와 관계")

# 데이터 불러오기 및 전처리
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"
    df = pd.read_csv(url)
    
    # 장르 전처리: 첫 번째 장르만 추출
    df['genre'] = df['genre'].astype(str).str.split('|').str[0]
    
    # 중복 에러 방지 및 결측치 처리
    df = df.dropna(subset=['movieCd', 'movieNm', 'genre', 'nation', 'total_audi', 'first_scrn', 'first_week_audi', 'days_in_top10'])
    df = df.drop_duplicates(subset=['movieCd'])
    
    return df

df = load_data()

st.divider()

# 첫 번째 구역: 장르별 영화 편수 (도넛 그래프)
st.subheader("1. 장르별 영화 편수 분포")

# 장르별 영화 편수 집계
genre_counts = df['genre'].value_counts().reset_index()
genre_counts.columns = ['genre', 'count']

# Plotly 도넛 그래프 생성
fig1 = px.pie(
    genre_counts,
    names='genre',
    values='count',
    hole=0.4,
    title="장르별 영화 비율 및 편수"
)

# 마우스오버 시 편수(value)와 비율(percent) 표시 설정
fig1.update_traces(
    textinfo='percent+label',
    hovertemplate="<b>장르: %{label}</b><br>편수: %{value}편<br>비율: %{percent}<extra></extra>"
)

# 그래프 출력
st.plotly_chart(fig1, use_container_width=True)

# 설명 구역
st.info("💡 **이 그래프로 알 수 있는 것:** 특정 장르가 전체 박스오피스 상위권 영화 중 가장 큰 비중을 차지하고 있으며, 장르별 편수 편차를 한눈에 비교할 수 있습니다.")

st.divider()

# 두 번째 구역: 장르 및 영화별 총 관객수 분포 (트리맵)
st.subheader("2. 장르별·영화별 총 관객수 (트리맵)")

# Plotly 트리맵 그래프 생성 (장르 -> 영화명 계층 구조)
fig2 = px.treemap(
    df,
    path=[px.Constant("전체 영화"), 'genre', 'movieNm'],
    values='total_audi',
    title="장르 및 영화별 총 관객수 트리맵"
)

# 마우스오버 시 영화명과 총 관객수 표시 설정
fig2.update_traces(
    hovertemplate="<b>%{label}</b><br>총 관객수: %{value:,}명<extra></extra>"
)

# 그래프 출력
st.plotly_chart(fig2, use_container_width=True)

# 설명 구역
st.info("💡 **이 그래프로 알 수 있는 것:** 장르 전체의 총 관객 규모뿐만 아니라 해당 장르 내에서 어떤 영화가 흥행을 주도했는지 직관적으로 비교할 수 있습니다.")

st.divider()

# 세 번째 구역: 총 관객수 분포 (히스토그램)
st.subheader("3. 총 관객수 분포")

# Plotly 히스토그램 생성
fig3 = px.histogram(
    df,
    x='total_audi',
    nbins=20,
    labels={'total_audi': '총 관객수 (명)'},
    title="영화별 총 관객수 분포 (히스토그램)"
)

fig3.update_traces(
    hovertemplate="관객수 구간: %{x}<br>영화 수: %{y}편<extra></extra>"
)

# 그래프 출력
st.plotly_chart(fig3, use_container_width=True)

# 가장 관객 수가 많은 영화 정보 추출
top_movie = df.loc[df['total_audi'].idxmax()]
top_movie_name = top_movie['movieNm']
top_movie_audi = top_movie['total_audi']

# 설명 구역 (분포 밀집 구간 및 최다 관객 영화 표기)
st.info(
    f"💡 **이 그래프로 알 수 있는 것:** 대부분의 영화는 관객수 하위 구간에 집중되어 분포하고 있으며, "
    f"가장 많은 관객을 동원한 영화는 **'{top_movie_name}'** (총 관객수: {top_movie_audi:,}명)입니다."
)

st.divider()

# 네 번째 구역: 개봉일 스크린수와 총 관객수의 관계 (산점도)
st.subheader("4. 개봉일 스크린수와 총 관객수의 관계")

# Plotly 산점도 생성 (장르별 색상 구분, hover_name에 영화명 지정)
fig4 = px.scatter(
    df,
    x='first_scrn',
    y='total_audi',
    color='genre',
    hover_name='movieNm',
    labels={
        'first_scrn': '개봉일 스크린수 (개)',
        'total_audi': '총 관객수 (명)',
        'genre': '장르'
    },
    title="개봉일 스크린수 vs 총 관객수 산점도"
)

# 마우스오버 표시 형식 설정
fig4.update_traces(
    hovertemplate="<b>%{hovertext}</b><br>개봉일 스크린수: %{x:,}개<br>총 관객수: %{y:,}명<extra></extra>"
)

# 그래프 출력
st.plotly_chart(fig4, use_container_width=True)

# 설명 구역
st.info("💡 **이 그래프로 알 수 있는 것:** 개봉일 스크린수가 많을수록 총 관객수도 대체로 증가하는 양의 상관관계를 보이나, 일부 영화는 스크린수에 비해 높은 관객 동원력을 기록했음을 알 수 있습니다.")

st.divider()

# 다섯 번째 구역: 영화 수 10편 이상 장르의 총 관객수 분포 (박스플롯)
st.subheader("5. 주요 장르별 총 관객수 분포 (박스플롯)")

# 10편 이상인 장르 필터링
genre_counts_series = df['genre'].value_counts()
top_genres = genre_counts_series[genre_counts_series >= 10].index
df_filtered = df[df['genre'].isin(top_genres)]

# Plotly 박스플롯 생성 (points='outliers'로 이상치 점 표시, hover_name 지정)
fig5 = px.box(
    df_filtered,
    x='genre',
    y='total_audi',
    color='genre',
    hover_name='movieNm',
    points='outliers',
    labels={
        'genre': '장르',
        'total_audi': '총 관객수 (명)'
    },
    title="영화 수 10편 이상 장르의 총 관객수 박스플롯"
)

# 마우스 오버 시 영화명, 장르, 총 관객수가 보이도록 설정
fig5.update_traces(
    hovertemplate="<b>%{hovertext}</b><br>장르: %{x}<br>총 관객수: %{y:,}명<extra></extra>"
)

# 그래프 출력
st.plotly_chart(fig5, use_container_width=True)

# 설명 구역
st.info("💡 **이 그래프로 알 수 있는 것:** 영화 편수가 10편 이상인 장르 간의 중간 관객수 분포 차이를 비교할 수 있으며, 박스 상단 밖으로 튀어나온 이상치 점을 통해 해당 장르 내의 기록적인 대흥행작을 한눈에 파악할 수 있습니다.")

st.divider()

# 여섯 번째 구역: 개봉일 스크린수, 총 관객수, 첫 주 관객수의 관계 (버블 차트)
st.subheader("6. 개봉일 스크린수·총 관객수·첫 주 관객수 관계 (버블 차트)")

# Plotly 버블 차트 생성 (size에 first_week_audi 지정)
fig6 = px.scatter(
    df,
    x='first_scrn',
    y='total_audi',
    size='first_week_audi',
    color='genre',
    hover_name='movieNm',
    size_max=50,
    labels={
        'first_scrn': '개봉일 스크린수 (개)',
        'total_audi': '총 관객수 (명)',
        'first_week_audi': '첫 주 관객수 (명)',
        'genre': '장르'
    },
    title="개봉일 스크린수 vs 총 관객수 (점 크기: 첫 주 관객수)"
)

# 마우스오버 표시 형식 설정
fig6.update_traces(
    hovertemplate="<b>%{hovertext}</b><br>개봉일 스크린수: %{x:,}개<br>총 관객수: %{y:,}명<br>첫 주 관객수: %{marker.size:,}명<extra></extra>"
)

# 그래프 출력
st.plotly_chart(fig6, use_container_width=True)

# 설명 구역
st.info("💡 **이 그래프로 알 수 있는 것:** 개봉일 스크린수와 총 관객수뿐만 아니라 점의 크기(첫 주 관객수)를 통해 초반 흥행 기세가 최종 총 관객수 형성에 얼마나 큰 영향을 주었는지 3가지 차원으로 함께 비교해 볼 수 있습니다.")

st.divider()

# 일곱 번째 구역: 제작 국가별 및 장르별 영화 편수 (선버스트 차트)
st.subheader("7. 제작 국가 및 장르별 영화 편수 (선버스트)")

# Plotly 선버스트 생성 (계층 구조: 국가 -> 장르, 칸 크기: 영화 편수)
fig7 = px.sunburst(
    df,
    path=['nation', 'genre'],
    title="제작 국가별 및 장르별 영화 편수 비율"
)

# 마우스오버 시 구분명, 영화 편수, 비율 표시 설정
fig7.update_traces(
    hovertemplate="<b>%{label}</b><br>영화 편수: %{value}편<br>비율: %{percentParent:.1%}<extra></extra>"
)

# 그래프 출력
st.plotly_chart(fig7, use_container_width=True)

# 설명 구역
st.info("💡 **이 그래프로 알 수 있는 것:** 주요 제작 국가별로 어떤 장르의 영화가 주로 제작 및 수입되었는지 계층 구조와 비율을 한눈에 파악할 수 있습니다.")

st.divider()

# 여덟 번째 구역: 10위권 머문 날수와 총 관객수의 관계 (산점도)
st.subheader("8. 10위권 머문 날수와 총 관객수의 관계")

# Plotly 산점도 생성
fig8 = px.scatter(
    df,
    x='days_in_top10',
    y='total_audi',
    color='genre',
    hover_name='movieNm',
    labels={
        'days_in_top10': '10위권에 머문 날수 (일)',
        'total_audi': '총 관객수 (명)',
        'genre': '장르'
    },
    title="10위권 머문 날수 vs 총 관객수 산점도"
)

# 마우스오버 표시 형식 설정
fig8.update_traces(
    hovertemplate="<b>%{hovertext}</b><br>10위권 머문 날수: %{x}일<br>총 관객수: %{y:,}명<extra></extra>"
)

# 그래프 출력
st.plotly_chart(fig8, use_container_width=True)

# 설명 구역
st.info("💡 **이 그래프로 알 수 있는 것:** 10위권 순위에 오랫동안 머문 영화일수록 총 관객수가 비례해서 증가하는 강한 양의 상관관계를 보여주어, 장기 흥행 여부가 최종 흥행 성패를 가르는 핵심 요소임을 알 수 있습니다.")
