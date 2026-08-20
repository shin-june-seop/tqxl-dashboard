import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from supabase import create_client, Client

# 페이지 기본 설정
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

# 에러 없는 안전한 시장 데이터 로더
@st.cache_data(ttl=3600)
def get_market_data(ticker):
    try:
        # 데이터 수집
        data = yf.Ticker(ticker)
        df = data.history(period="2y")
        
        if df.empty:
            df = yf.download(ticker, period="2y")
            
        if 'Close' in df.columns:
            if isinstance(df['Close'], pd.DataFrame):
                close_series = df['Close'].iloc[:, 0]
            else:
                close_series = df['Close']
        else:
            close_series = df.iloc[:, 0]
            
        df_res = pd.DataFrame({'Close': close_series})
        df_res['200MA'] = df_res['Close'].rolling(window=200).mean()
        df_res['Disparity_%'] = ((df_res['Close'] - df_res['200MA']) / df_res['200MA']) * 100
        return df_res
    except Exception as e:
        st.error(f"{ticker} 데이터를 불러오는 중 오류 발생: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_fx_rate():
    try:
        fx_data = yf.Ticker("USDKRW=X").history(period="5d")
        if not fx_data.empty and 'Close' in fx_data.columns:
            return float(fx_data['Close'].iloc[-1])
        return 1350.0 # 예비 기본값
    except:
        return 1350.0

# 사이드바 자산 선택
target_asset = st.sidebar.radio("분석 자산 선택", ["TQQQ (QQQ 기준)", "SOXL (SOXX 기준)"])

if "TQQQ" in target_asset:
    base_ticker, lev_ticker = "QQQ", "TQQQ"
    bubble_limit = 30.0
    emergency_1, emergency_2 = -15.0, -25.0
else:
    base_ticker, lev_ticker = "SOXX", "SOXL"
    bubble_limit = 40.0
    emergency_1, emergency_2 = -20.0, -30.0

df_base = get_market_data(base_ticker)
df_lev = get_market_data(lev_ticker)
usd_krw = get_fx_rate()

# 데이터 정상 확인 후 처리
if not df_base.empty and not df_lev.empty and len(df_base) >= 200:
    last_base_price = float(df_base['Close'].iloc[-1])
    last_base_200ma = float(df_base['200MA'].iloc[-1])
    base_disparity = float(df_base['Disparity_%'].iloc[-1])

    last_lev_price = float(df_lev['Close'].iloc[-1])
    lev_disparity = float(df_lev['Disparity_%'].iloc[-1])

    tab1, tab2, tab3, tab4 = st.tabs(["🚨 실시간 시그널", "📝 모바일 매매일지", "📈 차트 오버레이", "📜 전략 규칙"])

    # [TAB 1] 실시간 시그널
    with tab1:
        st.subheader(f"📊 {base_ticker} & {lev_ticker} 시장 현황")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{base_ticker} 현재가", f"${last_base_price:.2f}")
        col2.metric(f"{base_ticker} 200일선", f"${last_base_200ma:.2f}")
        col3.metric(f"{base_ticker} 이격도", f"{base_disparity:+.2f}%")
        col4.metric("현재 환율", f"{usd_krw:,.1f}원")

        st.markdown("---")
        
        is_emergency_2 = base_disparity <= emergency_2
        is_emergency_1 = base_disparity <= emergency_1 and not is_emergency_2
        is_bubble = lev_disparity >= bubble_limit

        if is_emergency_2:
            stage = f"🚨 2차 비상 대응 (기초지수 ≤ {emergency_2}%)"
            stock_ratio = 10 if lev_ticker == "SOXL" else 15
            cash_ratio = 90 if lev_ticker == "SOXL" else 85
            action_msg = f"월중 즉시 {lev_ticker} 비중을 {stock_ratio}%로 축소하고 현금 {cash_ratio}%를 확보하세요!"
            alert_type = "error"
        elif is_emergency_1:
            stage = f"⚠️ 1차 비상 대응 (기초지수 ≤ {emergency_1}%)"
            stock_ratio = 20 if lev_ticker == "SOXL" else 30
            cash_ratio = 80 if lev_ticker == "SOXL" else 70
            action_msg = f"월중 즉시 {lev_ticker} 비중을 {stock_ratio}%로 축소하고 현금 {cash_ratio}%를 확보하세요!"
            alert_type = "warning"
        elif is_bubble:
            stage = f"🎈 버블 보험 작동 ({lev_ticker} 이격도 ≥ +{bubble_limit}%)"
            stock_ratio, cash_ratio = 60, 40
            action_msg = f"{lev_ticker} 과열 상태입니다. 주식 비중을 60%로 낮추고 현금 40%를 확보하세요."
            alert_type = "warning"
        else:
            if base_disparity >= 0:
                stage, stock_ratio, cash_ratio = "1단계 (강세장)", 80, 20
                action_msg = "상승장 유지 중. 주식 80% / 현금 20% 비중을 유지하세요."
                alert_type = "success"
            elif -12.0 <= base_disparity < 0:
                stage = "2단계 (초기 하락)"
                stock_ratio = 40 if lev_ticker == "SOXL" else 60
                cash_ratio = 60 if lev_ticker == "SOXL" else 40
                action_msg = "200일선 이탈 초기. 1차 현금 확보를 진행하세요."
                alert_type = "info"
            else:
                stage = "3단계 (진성 하락)"
                stock_ratio = 20 if lev_ticker == "SOXL" else 30
                cash_ratio = 80 if lev_ticker == "SOXL" else 70
                action_msg = "대하락장 진입. 현금 쿠션으로 계좌를 보호하세요."
                alert_type = "warning"

        st.subheader("🎯 현재 적용 알림 (ACTION ALERT)")
        getattr(st, alert_type)(f"**현재 상태:** {stage}\n\n👉 **행동 요령:** {action_msg}")

    # [TAB 2] 모바일 매매일지
    with tab2:
        st.subheader("📲 매매일지 입력 (Supabase DB 저장)")
        
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
                supabase.table("trade_logs").insert(data).execute()
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
            st.warning("매매일지 데이터를 불러오는 중입니다.")

    # [TAB 3] 차트 오버레이
    with tab3:
        st.subheader(f"📈 {base_ticker} 200일선 차트")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_base.index, y=df_base['Close'], name=f'{base_ticker} 주가', line=dict(color='white')))
        fig.add_trace(go.Scatter(x=df_base.index, y=df_base['200MA'], name=f'{base_ticker} 200일선', line=dict(color='yellow', dash='dash')))
        fig.update_layout(height=500, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # [TAB 4] 전략 규칙
    with tab4:
        st.markdown("### 📜 TQQQ & SOXL 200일선 매트릭스 전략 규칙")
        st.info("실시간 이격도 분석 기반 현금 비중 리밸런싱 시스템 적용 완료")

else:
    st.warning("⚠️ 야후 파이낸스에서 데이터를 로딩 중이거나 불러오지 못했습니다. 잠시 후 페이지를 새로고침(F5)해 주세요.")
