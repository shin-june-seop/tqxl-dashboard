import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None

APP_VERSION = "V1"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
TRADES_FILE = DATA_DIR / "trades.csv"
ACCOUNT_FILE = DATA_DIR / "account.csv"
SNAPSHOT_FILE = DATA_DIR / "snapshots.csv"

st.set_page_config(page_title="TQQQ / SOXL Control Center V1", page_icon="📈", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
.hero {padding: 1.1rem 1.3rem; border-radius: 16px; background: linear-gradient(135deg,#101827,#182338); border:1px solid #29364e;}
.hero h1 {margin:0; font-size:2rem;}
.hero p {margin:.35rem 0 0; color:#aab7ca;}
.card {padding:1rem 1.05rem; border-radius:14px; border:1px solid #29364e; background:#101722; min-height:130px;}
.card-title {color:#9eabc0; font-size:.82rem; text-transform:uppercase; letter-spacing:.04em;}
.big {font-size:1.65rem; font-weight:700; margin-top:.25rem;}
.good {color:#31d17c;} .warn {color:#f2b84b;} .bad {color:#ff6b6b;} .info {color:#6fa8ff;}
.action {padding:1rem 1.1rem; border-radius:14px; border:1px solid #3c557a; background:#111d30;}
.action h3 {margin-top:0; color:#ffffff;}
.action p {color:#f1f5f9; font-size:1.02rem; line-height:1.55;}
.action .small {color:#d7e0ec;}
.stage {padding:.65rem .8rem; border-radius:10px; background:#151f2e; border:1px solid #29364e;}
.small {font-size:.82rem; color:#98a6ba;}
</style>
""", unsafe_allow_html=True,
)

# ---------- Strategy ----------
RULES = {
    "TQQQ": {
        "market": "QQQ", "bubble": 30, "emergency1": -15, "emergency2": -25,
        "stages": [(1, "강세장", 80, 20), (2, "초기 하락", 60, 40), (3, "전성 하락", 30, 70), (4, "시발", 15, 85), (5, "상승 복귀", 60, 40)],
        "stage3_text": "월말 200일선 2개월 연속 이탈 또는 200MA 대비 -12~-15%",
        "emergency_text": "QQQ가 200MA 대비 -15% 도달 시 월중 즉시 30%, -25% 도달 시 15%",
    },
    "SOXL": {
        "market": "SOXX", "bubble": 40, "emergency1": -20, "emergency2": -30,
        "stages": [(1, "강세장", 80, 20), (2, "초기 하락", 40, 60), (3, "전성 하락", 20, 80), (4, "시발", 10, 90), (5, "상승 복귀", 60, 40)],
        "stage3_text": "월말 200일선 1개월 연속 이탈 또는 200MA 대비 -15%",
        "emergency_text": "SOXX가 200MA 대비 -20% 도달 시 월중 즉시 20%, -30% 도달 시 10%",
    },
}


def load_csv(path, columns):
    if path.exists():
        try:
            df = pd.read_csv(path)
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=columns)


def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def fetch_history(ticker, period="3y", interval="1d"):
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={c: c.title() for c in df.columns})
        df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) else pd.to_datetime(df.index)
        return df.dropna(subset=["Close"])
    except Exception as e:
        st.warning(f"{ticker} 데이터 조회 실패: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def get_market_data():
    result = {}
    for t in ["QQQ", "SOXX", "TQQQ", "SOXL"]:
        result[t] = fetch_history(t, "3y", "1d")
    return result


def metrics_for(df):
    if df.empty:
        return None
    s = df["Close"].astype(float)
    dma = s.rolling(200).mean()
    return {
        "price": float(s.iloc[-1]),
        "dma": float(dma.iloc[-1]) if pd.notna(dma.iloc[-1]) else None,
        "dist": float((s.iloc[-1] / dma.iloc[-1] - 1) * 100) if pd.notna(dma.iloc[-1]) else None,
        "date": s.index[-1].date(),
    }


def monthly_status(df):
    """월말 규칙용 상태. 진행 중인 현재 월은 제외하고 완료된 월만 판단한다."""
    if df.empty or len(df) < 200:
        return {"current_above": None, "prev_above": None, "below_count": 0, "above_count": 0, "months": pd.DataFrame()}
    d = df[["Close"]].copy()
    d["dma200"] = d["Close"].rolling(200).mean()
    d = d.dropna()
    if d.empty:
        return {"current_above": None, "prev_above": None, "below_count": 0, "above_count": 0, "months": pd.DataFrame()}

    # 현재 월이 아직 끝나지 않았다면 월말 판정에서 제외한다.
    month_end = d.resample("ME").last().dropna()
    last_data_date = d.index[-1].date()
    if month_end.index[-1].date().month == last_data_date.month and month_end.index[-1].date().year == last_data_date.year:
        month_end = month_end.iloc[:-1]
    m = month_end.copy()
    if m.empty:
        return {"current_above": None, "prev_above": None, "below_count": 0, "above_count": 0, "months": m}
    m["above"] = m["Close"] > m["dma200"]

    below_count = 0
    for v in reversed(m["above"].tolist()):
        if not v:
            below_count += 1
        else:
            break
    above_count = 0
    for v in reversed(m["above"].tolist()):
        if v:
            above_count += 1
        else:
            break
    return {
        "current_above": bool(m["above"].iloc[-1]),
        "prev_above": bool(m["above"].iloc[-2]) if len(m) >= 2 else None,
        "below_count": below_count,
        "above_count": above_count,
        "months": m.tail(12),
    }


def recovery_status(df):
    """상승 복귀: 최근 10거래일(약 2주) 연속 200MA 위 + 그 직전에는 200MA 아래였는지 확인."""
    if df.empty or len(df) < 220:
        return {"qualified": False, "days_above": 0, "recent_reclaim": False}
    s = df["Close"].astype(float)
    dma = s.rolling(200).mean()
    valid = pd.DataFrame({"close": s, "dma": dma}).dropna()
    if len(valid) < 25:
        return {"qualified": False, "days_above": 0, "recent_reclaim": False}

    last10 = valid.tail(10)
    days_above = int((last10["close"] > last10["dma"]).sum())
    recent_reclaim = bool((valid.iloc[-20:-10]["close"] <= valid.iloc[-20:-10]["dma"]).any())
    return {
        "qualified": bool(days_above == 10 and recent_reclaim),
        "days_above": days_above,
        "recent_reclaim": recent_reclaim,
    }


def determine(ticker, market_df, etf_df):
    r = RULES[ticker]
    mm = metrics_for(market_df)
    em = metrics_for(etf_df)
    ms = monthly_status(market_df)
    recovery = recovery_status(market_df)
    stage = 1
    reason = ""
    action = "현재 목표 비중 유지. 정상 점검 주기(2주)를 따릅니다."
    emergency = None

    if mm and mm["dist"] is not None:
        if mm["dist"] <= r["emergency2"]:
            emergency = 2
        elif mm["dist"] <= r["emergency1"]:
            emergency = 1

    # 월말 기준 200MA 위 1개월 이상이면 정상 강세장(Stage 1)이 최우선이다.
    # Stage 5는 하락 후 재돌파하여 최근 10거래일(약 2주) 연속 위에 있는
    # "회복 과정"에서만 잠시 사용한다.
    if emergency == 2:
        stage = 4
        reason = f"월중 비상: {r['market']} 200MA 대비 {mm['dist']:.1f}% 이탈"
        action = f"월말까지 기다리지 않고 즉시 {ticker} 비중을 {r['stages'][3][2]}%로 축소, 현금 {r['stages'][3][3]}% 확보."
    elif emergency == 1:
        stage = 3
        reason = f"월중 비상: {r['market']} 200MA 대비 {mm['dist']:.1f}% 이탈"
        action = f"월말까지 기다리지 않고 즉시 {ticker} 비중을 {r['stages'][2][2]}%로 축소, 현금 {r['stages'][2][3]}% 확보."
    elif ms["current_above"] is False and ms["below_count"] >= (2 if ticker == "TQQQ" else 1):
        stage = 3
        reason = r["stage3_text"]
        action = f"{ticker} 비중을 {r['stages'][2][2]}%로 축소하고 현금 {r['stages'][2][3]}% 확보."
    elif ms["current_above"] is False and ms["prev_above"] is True:
        stage = 2
        reason = "월말 기준 200일선 첫 이탈"
        action = f"{ticker} 비중을 {r['stages'][1][2]}%로 축소하고 현금 {r['stages'][1][3]}% 확보."
    elif ms["above_count"] >= 1:
        stage = 1
        reason = "완료된 최근 월말 기준 200일선 위에서 1개월 이상 유지"
        action = f"{ticker} 목표 비중 {r['stages'][0][2]}% 유지. 과열도와 비상버튼만 감시."
    elif recovery["qualified"] and mm and mm["dist"] is not None and mm["dist"] > 0:
        stage = 5
        reason = "하락 후 200일선 재돌파 + 최근 10거래일(약 2주) 연속 200일선 위"
        action = "상승 복귀 단계: 목표 비중 60% / 현금 40%. 1개월 이상 강세장 확인 전까지 가짜 반등 여부를 확인합니다."
    elif ms["current_above"] is True:
        stage = 1
        reason = "월말 기준 200일선 위에서 강세장 유지"
        action = f"{ticker} 목표 비중 {r['stages'][0][2]}% 유지. 과열도와 비상버튼만 감시."

    bubble = em["dist"] is not None and em["dist"] >= r["bubble"]
    if bubble:
        action = f"⚠️ 버블보험 발동: {ticker} 비중을 60% 수준으로 낮추고 초과분을 현금화합니다."

    target = next(x[2] for x in r["stages"] if x[0] == stage)
    cash = 100 - target
    return {"stage": stage, "stage_name": next(x[1] for x in r["stages"] if x[0] == stage), "target": target, "cash": cash,
            "reason": reason, "action": action, "bubble": bubble, "market": mm, "etf": em, "monthly": ms, "recovery": recovery,
            "emergency": emergency}


def money(x):
    return f"${x:,.2f}"

# ---------- Sidebar ----------
with st.sidebar:
    st.title("📈 TQQQ / SOXL")
    page = st.radio("메뉴", ["Dashboard", "Accounts", "Trade Journal", "Performance", "Strategy / Rules", "Alerts", "Settings"])
    st.caption(f"Version {APP_VERSION}")
    if st.button("🔄 시장 데이터 새로고침"):
        st.cache_data.clear()
        st.rerun()

# ---------- Data ----------
market_data = get_market_data()
res_t = determine("TQQQ", market_data["QQQ"], market_data["TQQQ"])
res_s = determine("SOXL", market_data["SOXX"], market_data["SOXL"])

# ---------- Dashboard ----------
if page == "Dashboard":
    st.markdown('<div class="hero"><h1>Leveraged ETF Control Center</h1><p>TQQQ / SOXL 규칙 기반 운용 대시보드 · 2주 점검 + 월중 비상대응</p></div>', unsafe_allow_html=True)
    st.write("")

    # Top market cards
    c1, c2, c3, c4 = st.columns(4)
    for col, label, res in [(c1,"TQQQ / QQQ",res_t),(c2,"SOXL / SOXX",res_s)]:
        mm = res["market"]
        col.markdown(f'<div class="card"><div class="card-title">{label} 시장 상태</div><div class="big">STAGE {res["stage"]} · {res["stage_name"]}</div><div>{money(mm["price"]) if mm else "데이터 없음"} · 200MA {money(mm["dma"]) if mm and mm["dma"] else "-"}</div><div class="small">200MA 이격: <b>{mm["dist"]:+.2f}%</b> · 기준일 {mm["date"] if mm else "-"}</div></div>', unsafe_allow_html=True)
    c3.metric("TQQQ 목표 / 현금", f"{res_t['target']}% / {res_t['cash']}%")
    c4.metric("SOXL 목표 / 현금", f"{res_s['target']}% / {res_s['cash']}%")

    st.subheader("🎯 지금 해야 할 일")
    a1, a2 = st.columns(2)
    for col, ticker, res in [(a1,"TQQQ",res_t),(a2,"SOXL",res_s)]:
        tone = "bad" if res["bubble"] or res["emergency"] else ("warn" if res["stage"] in [2,3,4] else "good")
        badge = "🚨 비상" if res["emergency"] else ("⚠️ 버블보험" if res["bubble"] else "정상")
        col.markdown(f'<div class="action"><h3>{ticker} <span class="{tone}">{badge}</span></h3><div class="big">목표 {res["target"]}% / 현금 {res["cash"]}%</div><p>{res["action"]}</p><div class="small">판정 근거: {res["reason"]}</div><div class="small">완료 월말 200MA 위 연속: {res["monthly"]["above_count"]}개월 · 최근 2주 위: {res["recovery"]["days_above"]}/10거래일</div></div>', unsafe_allow_html=True)

    st.subheader("📊 전략 상태")
    rows=[]
    for t,res in [("TQQQ",res_t),("SOXL",res_s)]:
        rows.append({"ETF":t,"현재 단계":f"Stage {res['stage']} · {res['stage_name']}","목표 비중":f"{res['target']}%","현금":f"{res['cash']}%","버블보험":"발동" if res['bubble'] else "정상","2주 복귀":f"{res['recovery']['days_above']}/10일" if res['recovery']['days_above'] else "미충족"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("🚨 비상 조건 모니터")
    e1,e2=st.columns(2)
    e1.metric("QQQ 200MA 이격", f"{res_t['market']['dist']:+.2f}%" if res_t['market'] else "-")
    e1.caption("-15% → TQQQ 30% / -25% → 15%")
    e2.metric("SOXX 200MA 이격", f"{res_s['market']['dist']:+.2f}%" if res_s['market'] else "-")
    e2.caption("-20% → SOXL 20% / -30% → 10%")

    st.subheader("📈 최근 시장 흐름")
    chart_df = pd.DataFrame({"QQQ": market_data["QQQ"]["Close"], "SOXX": market_data["SOXX"]["Close"]}).dropna().tail(252)
    if not chart_df.empty:
        st.line_chart(chart_df)

    st.caption("※ V1은 규칙을 자동 계산해 행동을 제시하는 도구입니다. 실제 주문은 사용자가 확인 후 실행하세요.")

# ---------- Accounts ----------
elif page == "Accounts":
    st.title("💰 Accounts")
    st.caption("평단가·보유수량·현재가·평가액·수익률을 입력/관리합니다.")
    acct = load_csv(ACCOUNT_FILE, ["ETF","Shares","Avg Cost","Cash"])
    if acct.empty:
        acct = pd.DataFrame([{"ETF":"TQQQ","Shares":0.0,"Avg Cost":0.0,"Cash":0.0},{"ETF":"SOXL","Shares":0.0,"Avg Cost":0.0,"Cash":0.0}])
    edited = st.data_editor(acct, num_rows="fixed", use_container_width=True, key="account_editor")
    if st.button("💾 계좌 저장", type="primary"):
        save_csv(edited, ACCOUNT_FILE); st.success("계좌 정보가 저장되었습니다.")
    rows=[]
    total_value=0; total_cost=0
    for _,r in edited.iterrows():
        t=r["ETF"]; price=market_data.get(t,pd.DataFrame())
        cur=float(price["Close"].iloc[-1]) if not price.empty else 0
        shares=float(r["Shares"] or 0); avg=float(r["Avg Cost"] or 0); cash=float(r["Cash"] or 0)
        value=shares*cur; cost=shares*avg; pnl=value-cost
        total_value += value+cash; total_cost += cost
        rows.append({"ETF":t,"수량":shares,"평단":money(avg),"현재가":money(cur),"평가액":money(value),"손익":money(pnl),"수익률":f"{pnl/cost*100:+.2f}%" if cost else "-"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    m1,m2,m3=st.columns(3)
    m1.metric("총 평가액", money(total_value))
    m2.metric("총 투자원금", money(total_cost))
    m3.metric("총 손익", money(total_value-total_cost))

# ---------- Trade Journal ----------
elif page == "Trade Journal":
    st.title("📓 Trade Journal")
    st.caption("매매 이유와 당시 전략 단계를 함께 기록합니다.")
    trades=load_csv(TRADES_FILE,["Date","ETF","Side","Shares","Price","Amount","Stage","Reason","Memo"])
    with st.form("trade_form"):
        c=st.columns(5)
        date=c[0].date_input("날짜", datetime.now().date())
        etf=c[1].selectbox("ETF",["TQQQ","SOXL"])
        side=c[2].selectbox("구분",["매수","매도"])
        shares=c[3].number_input("수량",min_value=0.0,step=1.0)
        price=c[4].number_input("가격",min_value=0.0,step=0.01)
        stage=st.selectbox("당시 단계",["Stage 1","Stage 2","Stage 3","Stage 4","Stage 5","버블보험","월중 비상"])
        reason=st.text_input("매매 이유")
        memo=st.text_area("메모")
        submitted=st.form_submit_button("매매 기록 추가", type="primary")
    if submitted:
        new=pd.DataFrame([{"Date":str(date),"ETF":etf,"Side":side,"Shares":shares,"Price":price,"Amount":shares*price,"Stage":stage,"Reason":reason,"Memo":memo}])
        trades=pd.concat([trades,new],ignore_index=True); save_csv(trades,TRADES_FILE); st.success("매매가 기록되었습니다."); st.rerun()
    st.dataframe(trades.sort_values("Date",ascending=False) if not trades.empty else trades, use_container_width=True, hide_index=True)

# ---------- Performance ----------
elif page == "Performance":
    st.title("📊 Performance")
    acct=load_csv(ACCOUNT_FILE,["ETF","Shares","Avg Cost","Cash"])
    if acct.empty:
        st.info("Accounts 메뉴에서 보유수량과 평단가를 먼저 입력하세요.")
    else:
        perf=[]
        for _,r in acct.iterrows():
            t=r["ETF"]; df=market_data.get(t,pd.DataFrame())
            if df.empty: continue
            cur=float(df["Close"].iloc[-1]); avg=float(r["Avg Cost"]); shares=float(r["Shares"])
            pnl=(cur-avg)*shares; cost=avg*shares
            perf.append({"ETF":t,"수익률":pnl/cost*100 if cost else 0,"평가손익":pnl})
        if perf:
            p=pd.DataFrame(perf)
            st.dataframe(p, use_container_width=True, hide_index=True)
            st.bar_chart(p.set_index("ETF")["수익률"])

# ---------- Rules ----------
elif page == "Strategy / Rules":
    st.title("🛡️ Strategy / Rules")
    for ticker in ["TQQQ","SOXL"]:
        r=RULES[ticker]
        st.subheader(ticker)
        data=[]
        for n,name,target,cash in r["stages"]:
            if n==5: cond="200일선 재돌파 2주차"
            elif n==1: cond=f"{r['market']} > 200일선 (1개월 이상)"
            elif n==2: cond="200일선 첫 이탈"
            elif n==3: cond=r["stage3_text"]
            else: cond=f"{r['market']} 200일선 대비 -25% 이상 급락"
            data.append({"단계":f"Stage {n}","조건":cond,"{0} 비중".format(ticker):f"{target}%","현금":f"{cash}%"})
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        st.info(f"버블보험: {ticker} 자체 200MA 대비 +{r['bubble']}% 이상 → 주식 비중 60% 수준.\n\n월중 즉시: {r['emergency_text']}")

# ---------- Alerts ----------
elif page == "Alerts":
    st.title("🚨 Alerts")
    alerts=[]
    for t,res in [("TQQQ",res_t),("SOXL",res_s)]:
        if res["bubble"]: alerts.append({"종류":"버블보험","ETF":t,"상태":"HIGH","내용":f"자체 200MA 대비 +{res['etf']['dist']:.2f}% 이상"})
        if res["emergency"]==1: alerts.append({"종류":"월중 비상 1차","ETF":t,"상태":"HIGH","내용":"시장지수 200MA -1차 임계치 도달"})
        if res["emergency"]==2: alerts.append({"종류":"월중 비상 2차","ETF":t,"상태":"CRITICAL","내용":"시장지수 200MA -2차 임계치 도달"})
        if res["stage"] in [2,3,4]: alerts.append({"종류":"단계 변경","ETF":t,"상태":"ACTION","내용":res["action"]})
    if alerts: st.dataframe(pd.DataFrame(alerts),use_container_width=True,hide_index=True)
    else: st.success("현재 활성화된 경보가 없습니다.")

# ---------- Settings ----------
else:
    st.title("⚙️ Settings")
    st.write("V1에서는 전략 규칙을 코드에 고정해 실수로 변경되지 않도록 했습니다.")
    st.markdown("**데이터 저장 위치**")
    st.code(str(DATA_DIR))
    st.markdown("**점검 원칙**")
    st.write("정상 상태는 2주마다 점검하고, 월중 비상 임계치 도달 시 즉시 대응합니다.")
    st.warning("Streamlit Cloud에서 영구 저장이 필요하면 V2에서 Supabase 연결을 붙이는 것을 권장합니다.")
