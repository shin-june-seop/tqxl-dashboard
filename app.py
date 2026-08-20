import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from supabase import create_client, Client

# 페이지 설정
st.set_page_config(page_title="TQQQ & SOXL 자산배분 대시보드", layout="wide")

# Supabase 연동
@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

try:
    supabase = init_supabase()
except Exception as e:
    st.error("Supabase 연동 실패! Secrets 설정을 확인해 주세요.")

st.title("🛡️ TQQQ & SOXL 모바일 자산배분 대시보드")

# 시장 데이터 가져오기
@st.cache_data(ttl=3600)
def get_data(ticker):
    df = yf.download(ticker, period="2y")
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close'][[ticker]].rename(columns={ticker: 'Close'})
    else:
        df = df[['Close']]
    df['200MA'] = df['Close'].rolling(200).mean()
    df['Disparity_%'] = ((df['Close'] - df['200MA']) / df['200MA']) * 100
    return df

@st.cache_data(ttl=3600)
def get_fx():
    fx = yf.download("USDKRW=X", period="5d")
    if isinstance(fx.columns, pd.MultiIndex):
        return float(fx['Close'].iloc[-1].values[0])
    return float(fx['Close'].iloc[-1])

# 사이드바 자산 선택
target_asset = st.sidebar.radio("분석 자산 선택", ["TQQQ (QQQ 기준)", "SOXL (SOXX 기준)"])
base_ticker, lev_ticker = ("QQQ", "TQQQ") if "TQQQ" in target_asset else ("SOXX", "SOXL")

df_base = get_data(base_ticker)
df_lev = get_data(lev_ticker)
usd_krw = get_fx()

base_disparity = float(df_base['Disparity_%'].iloc[-1])
last_lev_price = float(df_lev['Close'].iloc[-1])

# 탭 구성
tab1, tab2, tab3 = st.tabs(["🚨 실시간 시그널", "📝 모바일 매매일지", "📜 전략 규칙"])

# [TAB 1] 실시간 시그널
with tab1:
    st.subheader(f"📊 {base_ticker} 200일선 이격도: {base_disparity:+.2f}%")
    st.metric("현재 환율", f"{usd_krw:,.1f} 원")
    
    if base_disparity >= 0:
        st.success("🟢 **1단계 (강세장):** 주식 80% / 현금 20% 유지")
    elif -15 <= base_disparity < 0:
        st.warning("🟡 **2단계 (초기하락):** 현금 비중 확대 진행")
    else:
        st.error("🔴 **비상 대응 단계:** 현금 쿠션 최대 확보!")

# [TAB 2] 모바일 매매일지 (Supabase 연동)
with tab2:
    st.subheader("📲 매매일지 입력 (DB 자동 저장)")
    
    with st.form("trade_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        t_ticker = col1.selectbox("종목", ["TQQQ", "SOXL"])
        t_type = col2.selectbox("구분", ["매수", "매도"])
        
        col3, col4, col5 = st.columns(3)
        t_qty = col3.number_input("수량", value=10, step=1)
        t_price = col4.number_input("단가 ($)", value=float(last_lev_price), step=0.1)
        t_fx = col5.number_input("환율 (원)", value=float(usd_krw), step=1.0)
        t_memo = st.text_input("메모", placeholder="매수 사유 등")
        
        btn = st.form_submit_button("DB에 저장하기")
        
        if btn:
            data = {
                "ticker": t_ticker,
                "trade_type": t_type,
                "qty": t_qty,
                "price": t_price,
                "fx_rate": t_fx,
                "memo": t_memo
            }
            res = supabase.table("trade_logs").insert(data).execute()
            st.success("✅ Supabase DB에 저장 완료!")

    st.markdown("---")
    st.subheader("📋 저장된 매매 기록")
    
    try:
        logs = supabase.table("trade_logs").select("*").order("created_at", desc=True).execute()
        if logs.data:
            df_logs = pd.DataFrame(logs.data)
            st.dataframe(df_logs[['created_at', 'ticker', 'trade_type', 'qty', 'price', 'fx_rate', 'memo']])
        else:
            st.info("아직 저장된 매매일지가 없습니다.")
    except Exception as e:
        st.warning("매매일지 데이터를 불러오는 중 오류가 발생했습니다.")

# [TAB 3] 전략 규칙
with tab3:
    st.markdown("### 💡 TQQQ / SOXL 200일선 매트릭스 전략 적용 중")