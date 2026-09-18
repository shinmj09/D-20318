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
    
    # 장르 전처리: Pandas str 메서드를 사용하여 첫 번째 장르만 안전하게 추출
    df['genre'] = df['genre'].astype(str).str.split('|').str[0]
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

# Plotly 트리맵 그래프 생성 (중복 방지를 위해 movieCd를 path에 사용)
fig2 = px.treemap(
    df,
    path=[px.Constant("전체"), 'genre', 'movieCd'],
    values='total_audi',
    hover_data={'movieNm': True, 'movieCd': False},
    title="장르 및 영화별 총 관객수 트리맵"
)

# 라벨을 영화명(movieNm)으로 변경하고 마우스오버 시 영화명과 총 관객수 표시
fig2.update_traces(
    textinfo='label',
    hovertemplate="<b>%{customdata[0]}</b><br>총 관객수: %{value:,}명<extra></extra>"
)

# 트리맵 라벨 텍스트 변경 (movieCd 대신 movieNm 표시)
fig2.data[0].labels = [
    df.set_index('movieCd')['movieNm'].to_dict().get(label, label)
    for label in fig2.data[0].labels
]

# 그래프 출력
st.plotly_chart(fig2, use_container_width=True)

# 설명 구역
st.info("💡 **이 그래프로 알 수 있는 것:** 장르 전체의 총 관객 규모뿐만 아니라 해당 장르 내에서 어떤 영화가 흥행을 주도했는지 직관적으로 비교할 수 있습니다.")
