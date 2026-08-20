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

# 시장 데이터 불러오기 함수
@st.cache_data(ttl=3600)
def get_market_data(ticker):
    try:
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
        st.error(f"{ticker} 데이터 불러오기 실패: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_fx_rate():
    try:
        fx_data = yf.Ticker("USDKRW=X").history(period="5d")
        if not fx_data.empty and 'Close' in fx_data.columns:
            return float(fx_data['Close'].iloc[-1])
        return 1350.0
    except:
        return 1350.0

# 기본 데이터 로드
df_qqq = get_market_data("QQQ")
df_tqqq = get_market_data("TQQQ")
df_soxx = get_market_data("SOXX")
df_soxl = get_market_data("SOXL")
usd_krw = get_fx_rate()

# 사이드바 자산 선택
target_asset = st.sidebar.radio("분석 자산 선택", ["TQQQ (QQQ 기준)", "SOXL (SOXX 기준)"])

if "TQQQ" in target_asset:
    base_ticker, lev_ticker = "QQQ", "TQQQ"
    df_base, df_lev = df_qqq, df_tqqq
    bubble_limit = 30.0
    emergency_1, emergency_2 = -15.0, -25.0
else:
    base_ticker, lev_ticker = "SOXX", "SOXL"
    df_base, df_lev = df_soxx, df_soxl
    bubble_limit = 40.0
    emergency_1, emergency_2 = -20.0, -30.0

# 탭 구성
tab1, tab2, tab3, tab4 = st.tabs(["📊 내 포트폴리오 & 시그널", "📝 모바일 매매일지", "📈 차트 오버레이", "📜 상세 전략 규칙"])

# [TAB 1] 포트폴리오 밸류 & 실시간 시그널
with tab1:
    st.subheader("💰 내 포트폴리오 자산 현황")
    
    tqqq_price = float(df_tqqq['Close'].iloc[-1]) if not df_tqqq.empty else 0
    soxl_price = float(df_soxl['Close'].iloc[-1]) if not df_soxl.empty else 0
    
    try:
        logs = supabase.table("trade_logs").select("*").execute()
        if logs.data:
            df_logs = pd.DataFrame(logs.data)
            
            summary_list = []
            total_eval_krw, total_buy_krw = 0, 0
            
            for tk, cur_p in [("TQQQ", tqqq_price), ("SOXL", soxl_price)]:
                df_tk = df_logs[df_logs['ticker'] == tk]
                if not df_tk.empty:
                    buys = df_tk[df_tk['trade_type'] == '매수']
                    sells = df_tk[df_tk['trade_type'] == '매도']
                    
                    buy_qty = buys['qty'].sum() if not buys.empty else 0
                    sell_qty = sells['qty'].sum() if not sells.empty else 0
                    hold_qty = buy_qty - sell_qty
                    
                    if hold_qty > 0:
                        avg_price = (buys['qty'] * buys['price']).sum() / buy_qty if buy_qty > 0 else 0
                        avg_fx = (buys['qty'] * buys['fx_rate']).sum() / buy_qty if buy_qty > 0 else usd_krw
                        
                        buy_val_krw = hold_qty * avg_price * avg_fx
                        eval_val_krw = hold_qty * cur_p * usd_krw
                        profit_krw = eval_val_krw - buy_val_krw
                        return_pct = (profit_krw / buy_val_krw) * 100 if buy_val_krw > 0 else 0
                        
                        total_buy_krw += buy_val_krw
                        total_eval_krw += eval_val_krw
                        
                        summary_list.append({
                            "종목": tk, "보유수량": f"{hold_qty:,} 주",
                            "평단가": f"${avg_price:.2f}", "현재가": f"${cur_p:.2f}",
                            "평가금액(원)": f"{eval_val_krw:,.0f}원",
                            "수익금(원)": f"{profit_krw:+,.0f}원",
                            "수익률": f"{return_pct:+.2f}%"
                        })
            
            total_profit_krw = total_eval_krw - total_buy_krw
            total_return_pct = (total_profit_krw / total_buy_krw) * 100 if total_buy_krw > 0 else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("총 평가 자산 (원)", f"{total_eval_krw:,.0f}원")
            c2.metric("총 매수 원금 (원)", f"{total_buy_krw:,.0f}원")
            c3.metric("총 평가 손익 (원)", f"{total_profit_krw:+,.0f}원", f"{total_return_pct:+.2f}%")
            c4.metric("적용 환율 (USD/KRW)", f"{usd_krw:,.1f}원")
            
            if summary_list:
                st.table(pd.DataFrame(summary_list))
        else:
            st.info("💡 매매일지에 기록을 입력하시면 실시간 보유 자산, 평가금액, 수익률이 자동으로 상단에 표시됩니다.")
    except Exception as e:
        st.warning("포트폴리오 평가액 계산 중 데이터 확인 필요")

    st.markdown("---")
    
    if not df_base.empty and not df_lev.empty and len(df_base) >= 200:
        last_base_price = float(df_base['Close'].iloc[-1])
        last_base_200ma = float(df_base['200MA'].iloc[-1])
        base_disparity = float(df_base['Disparity_%'].iloc[-1])

        last_lev_price = float(df_lev['Close'].iloc[-1])
        lev_disparity = float(df_lev['Disparity_%'].iloc[-1])

        st.subheader(f"🚨 실시간 시그널 ({base_ticker} & {lev_ticker})")
        
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric(f"{base_ticker} 현재가", f"${last_base_price:.2f}")
        sc2.metric(f"{base_ticker} 200일선", f"${last_base_200ma:.2f}")
        sc3.metric(f"{base_ticker} 200일선 이격도", f"{base_disparity:+.2f}%")

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
        t_price = col4.number_input("단가 ($)", value=float(tqqq_price if t_ticker=="TQQQ" else soxl_price), step=0.1)
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

# [TAB 3] 차트 오버레이 (선 색상 및 스타일 시각적 명확화)
with tab3:
    st.subheader(f"📈 {base_ticker} & {lev_ticker} 200일선 차트")
    if not df_base.empty:
        fig = go.Figure()
        
        # 1. 주가 선 (밝은 청록색, 선 두께 2)
        fig.add_trace(go.Scatter(
            x=df_base.index, 
            y=df_base['Close'], 
            name=f'{base_ticker} 주가', 
            line=dict(color='#00FFFF', width=2)
        ))
        
        # 2. 200일 이동평균선 (선명한 주황색, 점선 스타일)
        fig.add_trace(go.Scatter(
            x=df_base.index, 
            y=df_base['200MA'], 
            name=f'{base_ticker} 200일선', 
            line=dict(color='#FF9900', width=2, dash='dash')
        ))
        
        fig.update_layout(
            height=500, 
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

# [TAB 4] 상세 전략 규칙 안내
with tab4:
    st.subheader("📜 TQQQ & SOXL 단계별 동적 자산배분 전략 세부 규칙")
    
    st.markdown("### 1️⃣ 단계별 비중 매트릭스 (2주 주기)")
    
    col_tqqq, col_soxl = st.columns(2)
    
    with col_tqqq:
        st.markdown("#### 🟦 TQQQ 전략 (QQQ 위치 기준)")
        st.table(pd.DataFrame({
            "구분": ["1단계 (강세장)", "2단계 (초기하락)", "3단계 (진성하락)", "4단계 (시발)", "🔄 상승복귀"],
            "시장 상태 (QQQ 위치)": ["QQQ > 200일선 (1개월 이상)", "200일선 첫 이탈 OR 2개월 연속 이탈", "200일선 대비 -12% ~ -15% 이탈", "200일선 대비 -25% 이상 급락", "200일선 재돌파 2주 차"],
            "TQQQ 비중": ["80%", "60%", "30%", "15%", "60%"],
            "현금 비중": ["20%", "40%", "70%", "85%", "40%"],
            "핵심 행동 요령": ["상승장의 3배 수익 극대화", "1차 현금 확보 (단기 휩소 대비)", "대하락장 진입, 현금 쿠션으로 계좌 보호", "TQQQ 녹아내림 차단 & 최후 자산 보존", "가짜 반등 방지 (2개월 연속 시 80%)"]
        }))

    with col_soxl:
        st.markdown("#### 🟧 SOXL 전략 (SOXX 위치 기준)")
        st.table(pd.DataFrame({
            "구분": ["1단계 (강세장)", "2단계 (초기하락)", "3단계 (진성하락)", "4단계 (시발)", "🔄 상승복귀"],
            "시장 상태 (SOXX 위치)": ["SOXX > 200일선 (1개월 이상)", "200일선 첫 이탈 OR 2개월 연속 이탈", "200일선 대비 -15% 이탈", "200일선 대비 -25% 이상 급락", "200일선 재돌파 2주 차"],
            "SOXL 비중": ["80%", "40%", "20%", "10%", "60%"],
            "현금 비중": ["20%", "60%", "80%", "90%", "40%"],
            "핵심 행동 요령": ["상승장의 3배 수익 극대화", "1차 현금 확보 (단기 휩소 대비)", "대하락장 진입, 현금 쿠션으로 계좌 보호", "SOXL 녹아내림 차단 & 최후 자산 보존", "가짜 반등 방지 (2개월 연속 시 80%)"]
        }))

    st.markdown("---")
    st.markdown("### 2️⃣ 추가 특수 위험 관리 규칙")
    
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.info("🎈 **[버블 보험 (이격도 과열)]**\n\nTQQQ(또는 SOXL)의 현재 주가가 자기 자신의 200일 이동평균선보다 **+30%(SOXL은 +40%) 이상** 높게 치솟은 경우 (과열) ➔ **주식 비중 60%로 낮추기**")

    with col_r2:
        st.error("🚨 **[월중 즉시 대응 (비상 버튼)]**\n\n- **1차 비상 버튼:** 기초지수(QQQ/SOXX) ≤ 200MA -15%(-20%) 도달 시 ➔ 월중 즉시 **주식 비중 30%(20%)로 축소**\n- **2차 비상 버튼:** 기초지수(QQQ/SOXX) ≤ 200MA -25%(-30%) 도달 시 ➔ 월중 즉시 **주식 비중 15%(10%)로 축소**")
