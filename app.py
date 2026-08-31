import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = object

APP_VERSION = "V2.4"

st.set_page_config(page_title="TQQQ / SOXL Control Center V2", page_icon="📈", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
.hero {padding: 1.1rem 1.3rem; border-radius: 16px; background: linear-gradient(135deg,#101827,#182338); border:1px solid #29364e;}
.hero h1 {margin:0; font-size:2rem;}
.hero p {margin:.35rem 0 0; color:#dbe5f2;}
.card {padding:1rem 1.05rem; border-radius:14px; border:1px solid #29364e; background:#101722; min-height:130px;}
.card-title {color:#c8d3e2; font-size:.82rem; text-transform:uppercase; letter-spacing:.04em;}
.big {font-size:1.65rem; font-weight:700; margin-top:.25rem;}
.good {color:#31d17c;} .warn {color:#f2b84b;} .bad {color:#ff6b6b;} .info {color:#6fa8ff;}
.action {padding:1rem 1.1rem; border-radius:14px; border:1px solid #3c557a; background:#111d30;}
.action h3 {margin-top:0; color:#ffffff;}
.action p {color:#ffffff; font-size:1.02rem; line-height:1.55;}
.action .small {color:#e5edf7;}
.small {font-size:.82rem; color:#c9d4e2;}
.successbox {padding:1rem; border-radius:12px; border:1px solid #2d7650; background:#10251b; color:#f0fff6;}
.warnbox {padding:1rem; border-radius:12px; border:1px solid #8a6a2b; background:#292210; color:#fff7df;}
.errorbox {padding:1rem; border-radius:12px; border:1px solid #8d3b3b; background:#2b1515; color:#fff0f0;}
</style>
""", unsafe_allow_html=True,
)

# ---------- Supabase ----------
@st.cache_resource
def get_supabase():
    if create_client is None:
        return None, "supabase package is not installed."
    try:
        secrets = st.secrets
        url = secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
        key = secrets.get("SUPABASE_SERVICE_ROLE_KEY", secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_SERVICE_ROLE_KEY", os.getenv("SUPABASE_KEY", ""))))
    except Exception:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", os.getenv("SUPABASE_KEY", ""))
    if not url or not key:
        return None, "Supabase Secrets가 설정되지 않았습니다."
    try:
        return create_client(url, key), None
    except Exception as e:
        return None, f"Supabase 연결 실패: {e}"

supabase, sb_error = get_supabase()

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
        idx = pd.to_datetime(df.index)
        df.index = idx.tz_localize(None) if getattr(idx, "tz", None) else idx
        return df.dropna(subset=["Close"])
    except Exception as e:
        st.warning(f"{ticker} 데이터 조회 실패: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def get_market_data():
    return {t: fetch_history(t, "3y", "1d") for t in ["QQQ", "SOXX", "TQQQ", "SOXL"]}


def metrics_for(df):
    if df.empty:
        return None
    s = df["Close"].astype(float)
    dma = s.rolling(200).mean()
    if pd.isna(dma.iloc[-1]):
        return None
    return {"price": float(s.iloc[-1]), "dma": float(dma.iloc[-1]), "dist": float((s.iloc[-1] / dma.iloc[-1] - 1) * 100), "date": s.index[-1].date()}


def monthly_status(df):
    if df.empty or len(df) < 200:
        return {"current_above": None, "prev_above": None, "below_count": 0, "above_count": 0}
    d = df[["Close"]].copy()
    d["dma200"] = d["Close"].rolling(200).mean()
    d = d.dropna()
    if d.empty:
        return {"current_above": None, "prev_above": None, "below_count": 0, "above_count": 0}
    month_end = d.resample("ME").last().dropna()
    last_data_date = d.index[-1].date()
    if month_end.index[-1].date().month == last_data_date.month and month_end.index[-1].date().year == last_data_date.year:
        month_end = month_end.iloc[:-1]
    if month_end.empty:
        return {"current_above": None, "prev_above": None, "below_count": 0, "above_count": 0}
    month_end["above"] = month_end["Close"] > month_end["dma200"]
    below_count = 0
    for v in reversed(month_end["above"].tolist()):
        if not v: below_count += 1
        else: break
    above_count = 0
    for v in reversed(month_end["above"].tolist()):
        if v: above_count += 1
        else: break
    return {"current_above": bool(month_end["above"].iloc[-1]), "prev_above": bool(month_end["above"].iloc[-2]) if len(month_end) >= 2 else None, "below_count": below_count, "above_count": above_count}


def recovery_status(df):
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
    return {"qualified": bool(days_above == 10 and recent_reclaim), "days_above": days_above, "recent_reclaim": recent_reclaim}


def determine(ticker, market_df, etf_df):
    r = RULES[ticker]
    mm, em, ms, recovery = metrics_for(market_df), metrics_for(etf_df), monthly_status(market_df), recovery_status(market_df)
    stage, reason, action, emergency = 1, "", "현재 목표 비중 유지. 정상 점검 주기(2주)를 따릅니다.", None
    if mm:
        if mm["dist"] <= r["emergency2"]: emergency = 2
        elif mm["dist"] <= r["emergency1"]: emergency = 1
    if emergency == 2:
        stage, reason = 4, f"월중 비상: {r['market']} 200MA 대비 {mm['dist']:.1f}% 이탈"
        action = f"월말까지 기다리지 않고 즉시 {ticker} 비중을 {r['stages'][3][2]}%로 축소, 현금 {r['stages'][3][3]}% 확보."
    elif emergency == 1:
        stage, reason = 3, f"월중 비상: {r['market']} 200MA 대비 {mm['dist']:.1f}% 이탈"
        action = f"월말까지 기다리지 않고 즉시 {ticker} 비중을 {r['stages'][2][2]}%로 축소, 현금 {r['stages'][2][3]}% 확보."
    elif ms["current_above"] is False and ms["below_count"] >= (2 if ticker == "TQQQ" else 1):
        stage, reason = 3, r["stage3_text"]
        action = f"{ticker} 비중을 {r['stages'][2][2]}%로 축소하고 현금 {r['stages'][2][3]}% 확보."
    elif ms["current_above"] is False and ms["prev_above"] is True:
        stage, reason = 2, "완료된 월말 기준 200일선 첫 이탈"
        action = f"{ticker} 비중을 {r['stages'][1][2]}%로 축소하고 현금 {r['stages'][1][3]}% 확보."
    elif ms["above_count"] >= 1:
        stage, reason = 1, "완료된 최근 월말 기준 200일선 위에서 1개월 이상 유지"
        action = f"{ticker} 목표 비중 {r['stages'][0][2]}% 유지. 과열도와 비상버튼만 감시."
    elif recovery["qualified"] and mm and mm["dist"] > 0:
        stage, reason = 5, "하락 후 200일선 재돌파 + 최근 10거래일(약 2주) 연속 200일선 위"
        action = "상승 복귀 단계: 목표 비중 60% / 현금 40%. 완료된 월말 200MA 위 1개월 확인 전까지 가짜 반등 여부를 확인합니다."
    elif ms["current_above"] is True:
        stage, reason = 1, "최근 완료 월말 기준 200일선 위"
        action = f"{ticker} 목표 비중 {r['stages'][0][2]}% 유지."
    bubble = bool(em and em["dist"] >= r["bubble"])
    if bubble:
        action = f"⚠️ 버블보험 발동: {ticker} 비중을 60% 수준으로 낮추고 초과분을 현금화합니다."
    target = next(x[2] for x in r["stages"] if x[0] == stage)
    return {"stage": stage, "stage_name": next(x[1] for x in r["stages"] if x[0] == stage), "target": target, "cash": 100-target, "reason": reason, "action": action, "bubble": bubble, "market": mm, "etf": em, "monthly": ms, "recovery": recovery, "emergency": emergency}


def money(x): return f"${x:,.2f}"


def table_rows(data):
    return pd.DataFrame(data) if data else pd.DataFrame()

# ---------- DB helpers ----------
def db_select(table, columns="*", order=None):
    if supabase is None:
        return pd.DataFrame()
    try:
        q = supabase.table(table).select(columns)
        if order:
            q = q.order(order[0], desc=order[1])
        res = q.execute()
        return pd.DataFrame(res.data or [])
    except Exception as e:
        st.error(f"Supabase 조회 오류 ({table}): {e}")
        return pd.DataFrame()


def db_upsert(table, payload, on_conflict=None):
    if supabase is None:
        return False
    try:
        q = supabase.table(table).upsert(payload, on_conflict=on_conflict) if on_conflict else supabase.table(table).upsert(payload)
        q.execute()
        return True
    except Exception as e:
        st.error(f"Supabase 저장 오류 ({table}): {e}")
        return False


def db_insert(table, payload):
    if supabase is None:
        return False
    try:
        supabase.table(table).insert(payload).execute()
        return True
    except Exception as e:
        st.error(f"Supabase 입력 오류 ({table}): {e}")
        return False


def load_accounts():
    df = db_select("accounts")
    if df.empty:
        return pd.DataFrame([
            {"etf":"TQQQ","shares":0.0,"avg_cost":0.0,"cash":0.0,"initial_investment":0.0},
            {"etf":"SOXL","shares":0.0,"avg_cost":0.0,"cash":0.0,"initial_investment":0.0},
        ])
    for c in ["shares","avg_cost","cash","initial_investment"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else: df[c] = 0.0
    return df[["etf","shares","avg_cost","cash","initial_investment"]]


def account_summary(accounts, market_data):
    """Return per-ETF live account metrics based on initial investment."""
    rows = []
    total_initial = 0.0
    total_market_value = 0.0
    total_cash = 0.0
    total_assets = 0.0
    for _, r in accounts.iterrows():
        t = str(r["etf"])
        shares = float(r.get("shares", 0) or 0)
        avg = float(r.get("avg_cost", 0) or 0)
        cash = float(r.get("cash", 0) or 0)
        initial = float(r.get("initial_investment", 0) or 0)
        df = market_data.get(t, pd.DataFrame())
        cur = float(df["Close"].iloc[-1]) if not df.empty else 0.0
        cost = shares * avg
        value = shares * cur
        assets = value + cash
        pnl = assets - initial if initial > 0 else value - cost
        return_pct = (pnl / initial * 100) if initial > 0 else ((value - cost) / cost * 100 if cost else 0.0)
        total_initial += initial
        total_market_value += value
        total_cash += cash
        total_assets += assets
        rows.append({
            "ETF": t, "수량": shares, "평단": avg, "현재가": cur,
            "평가액": value, "계좌평가액": assets, "투자원금": initial,
            "매입원가": cost, "손익": pnl, "수익률": return_pct,
            "현금": cash,
        })
    total_pnl = total_assets - total_initial if total_initial > 0 else total_market_value - sum(float(r.get("매입원가",0) or 0) for r in rows)
    total_return = (total_pnl / total_initial * 100) if total_initial > 0 else 0.0
    return pd.DataFrame(rows), {
        "initial": total_initial, "market_value": total_market_value,
        "cash": total_cash, "assets": total_assets, "pnl": total_pnl,
        "return_pct": total_return,
    }


def trade_cashflow(etf):
    """Net cash flow generated by existing trades: sells add cash, buys consume cash."""
    trades = db_select("trades")
    if trades.empty or "etf" not in trades.columns:
        return 0.0
    t = trades[trades["etf"] == etf].copy()
    if t.empty:
        return 0.0
    amount = pd.to_numeric(t.get("amount"), errors="coerce").fillna(0.0)
    side = t.get("side", pd.Series(index=t.index, dtype=str))
    return float(amount.where(side == "매도", -amount).sum())


# ---------- Sidebar ----------
with st.sidebar:
    st.title("📈 TQQQ / SOXL")
    page = st.radio("메뉴", ["Dashboard", "Accounts", "Trade Journal", "Performance", "Strategy / Rules", "Alerts", "Settings"])
    st.caption(f"Version {APP_VERSION}")
    if supabase:
        st.success("Supabase 연결됨")
    else:
        st.error("Supabase 미연결")
    if st.button("🔄 시장 데이터 새로고침"):
        get_market_data.clear()
        st.rerun()

# ---------- Data ----------
market_data = get_market_data()
res_t = determine("TQQQ", market_data["QQQ"], market_data["TQQQ"])
res_s = determine("SOXL", market_data["SOXX"], market_data["SOXL"])

# ---------- Dashboard ----------
if page == "Dashboard":
    st.markdown('<div class="hero"><h1>Leveraged ETF Control Center</h1><p>TQQQ / SOXL 규칙 기반 운용 · 2주 점검 + 월중 비상대응 · Supabase 영구 저장</p></div>', unsafe_allow_html=True)
    if not supabase:
        st.warning("Supabase가 연결되지 않았습니다. Settings에서 Secrets를 설정하면 계좌/매매일지/스냅샷이 영구 저장됩니다.")

    # Live account overview
    accounts = load_accounts() if supabase else pd.DataFrame(columns=["etf","shares","avg_cost","cash","initial_investment"])
    acct_rows, totals = account_summary(accounts, market_data) if not accounts.empty else (pd.DataFrame(), {"invested":0,"market_value":0,"cash":0,"assets":0,"pnl":0,"return_pct":0})
    st.subheader("💰 내 계좌 현황")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("총 평가액", money(totals["assets"]))
    m2.metric("초기 투자금", money(totals["initial"]))
    m3.metric("평가손익", money(totals["pnl"]))
    m4.metric("수익률", f"{totals['return_pct']:+.2f}%" if totals["initial"] > 0 else "설정 필요")

    st.markdown("#### 💵 초기 투자금 설정")
    st.caption("TQQQ와 SOXL은 각각 별도 전략 계좌로 계산합니다. 초기 투자금을 저장하면 현재 실제 비중을 각 계좌의 총자산 기준으로 계산합니다.")
    ic1, ic2, ic3 = st.columns([1,1,1])
    initial_values = {}
    for col, ticker in [(ic1, "TQQQ"), (ic2, "SOXL")]:
        row = accounts[accounts["etf"] == ticker] if not accounts.empty else pd.DataFrame()
        current_initial = float(row["initial_investment"].iloc[0]) if not row.empty else 0.0
        initial_values[ticker] = col.number_input(f"{ticker} 초기 투자금 ($)", min_value=0.0, value=current_initial, step=100.0, key=f"initial_{ticker}")
    if ic3.button("💾 초기 투자금 저장", type="primary", use_container_width=True):
        payload=[]
        for ticker in ["TQQQ","SOXL"]:
            row = accounts[accounts["etf"] == ticker]
            shares = float(row["shares"].iloc[0]) if not row.empty else 0.0
            avg = float(row["avg_cost"].iloc[0]) if not row.empty else 0.0
            initial = float(initial_values[ticker])
            existing_cashflow = trade_cashflow(ticker)
            # If trades exist, preserve their net cash flow; otherwise assume current holdings came from the initial capital.
            new_cash = initial + existing_cashflow if existing_cashflow != 0 else initial - (shares * avg)
            payload.append({"etf":ticker,"initial_investment":initial,"cash":new_cash,"updated_at":datetime.utcnow().isoformat()})
        if db_upsert("accounts", payload, "etf"):
            st.success("TQQQ / SOXL 초기 투자금이 저장되었습니다. 실제 비중 계산도 갱신됩니다.")
            st.rerun()

    if not acct_rows.empty:
        display = acct_rows.copy()
        display["평단"] = display["평단"].map(money); display["현재가"] = display["현재가"].map(money)
        display["평가액"] = display["평가액"].map(money); display["손익"] = display["손익"].map(money)
        display["투자원금"] = display["투자원금"].map(money)
        display["수익률"] = display["수익률"].map(lambda x:f"{x:+.2f}%")
        st.dataframe(display[["ETF","수량","평단","현재가","평가액","투자원금","손익","수익률"]], use_container_width=True, hide_index=True)
        st.caption("※ Trade Journal에서 매수/매도를 저장하면 수량·평단·현금이 자동으로 Accounts에 반영됩니다.")

    c1, c2 = st.columns(2)
    for col, label, res in [(c1,"TQQQ / QQQ",res_t),(c2,"SOXL / SOXX",res_s)]:
        mm = res["market"]
        col.markdown(f'<div class="card"><div class="card-title">{label} 시장 상태</div><div class="big">STAGE {res["stage"]} · {res["stage_name"]}</div><div>{money(mm["price"]) if mm else "데이터 없음"} · 200MA {money(mm["dma"]) if mm else "-"}</div><div class="small">200MA 이격: <b>{mm["dist"]:+.2f}%</b> · 기준일 {mm["date"] if mm else "-"}</div></div>', unsafe_allow_html=True)

    st.subheader("🎯 지금 해야 할 일")
    a1, a2 = st.columns(2)
    for col, ticker, res in [(a1,"TQQQ",res_t),(a2,"SOXL",res_s)]:
        tone = "bad" if res["bubble"] or res["emergency"] else ("warn" if res["stage"] in [2,3,4] else "good")
        badge = "🚨 비상" if res["emergency"] else ("⚠️ 버블보험" if res["bubble"] else "정상")
        col.markdown(f'<div class="action"><h3>{ticker} <span class="{tone}">{badge}</span></h3><div class="big">목표 {res["target"]}% / 현금 {res["cash"]}%</div><p>{res["action"]}</p><div class="small">판정 근거: {res["reason"]}</div><div class="small">완료 월말 200MA 위 연속: {res["monthly"]["above_count"]}개월 · 최근 2주 위: {res["recovery"]["days_above"]}/10거래일</div></div>', unsafe_allow_html=True)

    st.subheader("📊 목표 비중 vs 실제 비중")
    alloc_rows=[]
    for t,res in [("TQQQ",res_t),("SOXL",res_s)]:
        row = acct_rows[acct_rows["ETF"]==t] if not acct_rows.empty else pd.DataFrame()
        value = float(row["평가액"].iloc[0]) if not row.empty else 0.0
        account_assets = float(row["계좌평가액"].iloc[0]) if not row.empty else 0.0
        initial = float(row["투자원금"].iloc[0]) if not row.empty else 0.0
        actual = (value / account_assets * 100) if account_assets > 0 else None
        alloc_rows.append({
            "ETF":t,
            "목표 비중":f"{res['target']}%",
            "실제 비중":f"{actual:.1f}%" if actual is not None else "투자금 설정 필요",
            "차이":f"{actual-res['target']:+.1f}%p" if actual is not None else "-",
            "계좌 평가액":money(account_assets),
        })
    st.dataframe(pd.DataFrame(alloc_rows), use_container_width=True, hide_index=True)
    st.caption("실제 비중 = 해당 ETF 평가액 ÷ 해당 ETF 계좌 총자산(ETF 평가액 + 현금) × 100")

    st.subheader("🚨 비상 조건 모니터")
    e1,e2=st.columns(2)
    e1.metric("QQQ 200MA 이격", f"{res_t['market']['dist']:+.2f}%" if res_t['market'] else "-")
    e1.caption("-15% → TQQQ 30% / -25% → 15%")
    e2.metric("SOXX 200MA 이격", f"{res_s['market']['dist']:+.2f}%" if res_s['market'] else "-")
    e2.caption("-20% → SOXL 20% / -30% → 10%")

    st.subheader("📈 최근 시장 흐름")
    chart_df = pd.DataFrame({"QQQ": market_data["QQQ"]["Close"], "SOXX": market_data["SOXX"]["Close"]}).dropna().tail(252)
    if not chart_df.empty: st.line_chart(chart_df)

    if supabase:
        if st.button("📸 오늘의 전략 스냅샷 저장", type="primary"):
            today = str(datetime.now().date())
            payload=[]
            for ticker,res in [("TQQQ",res_t),("SOXL",res_s)]:
                payload.append({"snapshot_date":today,"etf":ticker,"market_ticker":RULES[ticker]["market"],"market_price":res["market"]["price"] if res["market"] else None,"market_200ma":res["market"]["dma"] if res["market"] else None,"market_dist":res["market"]["dist"] if res["market"] else None,"etf_price":res["etf"]["price"] if res["etf"] else None,"etf_200ma":res["etf"]["dma"] if res["etf"] else None,"etf_dist":res["etf"]["dist"] if res["etf"] else None,"stage":res["stage"],"stage_name":res["stage_name"],"target_pct":res["target"],"cash_pct":res["cash"],"emergency":res["emergency"],"bubble":res["bubble"],"reason":res["reason"],"action":res["action"]})
            if db_upsert("snapshots",payload,"snapshot_date,etf"): st.success("오늘의 TQQQ / SOXL 전략 상태가 Supabase에 저장되었습니다.")
    st.caption("※ 실제 주문은 규칙과 계좌 상태를 확인한 후 사용자가 실행하세요.")

# ---------- Accounts ----------
elif page == "Accounts":
    st.title("💰 Accounts")
    st.caption("Supabase에 영구 저장되는 실제 계좌 현황입니다.")
    if supabase is None:
        st.error("Supabase 연결 후 사용할 수 있습니다.")
    else:
        acct = load_accounts()
        view = acct.rename(columns={"etf":"ETF","shares":"보유수량","avg_cost":"평단가","cash":"현금","initial_investment":"초기 투자금"})
        edited = st.data_editor(view, num_rows="fixed", use_container_width=True, key="account_editor", column_config={"ETF":st.column_config.TextColumn(disabled=True),"보유수량":st.column_config.NumberColumn(min_value=0.0),"평단가":st.column_config.NumberColumn(min_value=0.0,format="$%.2f"),"현금":st.column_config.NumberColumn(min_value=-100000000.0,format="$%.2f"),"초기 투자금":st.column_config.NumberColumn(min_value=0.0,format="$%.2f")})
        if st.button("💾 계좌 저장", type="primary"):
            payload=[]
            for _,r in edited.iterrows():
                payload.append({"etf":str(r["ETF"]),"shares":float(r["보유수량"] or 0),"avg_cost":float(r["평단가"] or 0),"cash":float(r["현금"] or 0),"initial_investment":float(r["초기 투자금"] or 0),"updated_at":datetime.utcnow().isoformat()})
            if db_upsert("accounts",payload,"etf"):
                st.success("계좌 정보가 Supabase에 저장되었습니다.")
                st.rerun()
        rows=[]; total_value=0; total_cost=0; total_cash=0
        for _,r in edited.iterrows():
            t=str(r["ETF"]); df=market_data.get(t,pd.DataFrame()); cur=float(df["Close"].iloc[-1]) if not df.empty else 0
            shares=float(r["보유수량"] or 0); avg=float(r["평단가"] or 0); cash=float(r["현금"] or 0)
            value=shares*cur; cost=shares*avg; pnl=value-cost
            total_value += value; total_cost += cost; total_cash += cash
            rows.append({"ETF":t,"수량":shares,"평단":money(avg),"현재가":money(cur),"평가액":money(value),"손익":money(pnl),"수익률":f"{pnl/cost*100:+.2f}%" if cost else "-","현금":money(cash)})
        st.dataframe(table_rows(rows), use_container_width=True, hide_index=True)
        m1,m2,m3,m4=st.columns(4)
        m1.metric("총 평가액", money(total_value+total_cash)); m2.metric("총 투자원금", money(total_cost)); m3.metric("총 손익", money(total_value-total_cost)); m4.metric("현금", money(total_cash))

# ---------- Trade Journal ----------
elif page == "Trade Journal":
    st.title("📓 Trade Journal")
    st.caption("매매 기록을 저장하면 Accounts의 수량·평단·현금이 자동으로 갱신됩니다.")
    if supabase is None:
        st.error("Supabase 연결 후 사용할 수 있습니다.")
    else:
        with st.form("trade_form"):
            c=st.columns(5)
            date=c[0].date_input("날짜", datetime.now().date()); etf=c[1].selectbox("ETF",["TQQQ","SOXL"]); side=c[2].selectbox("구분",["매수","매도"]); shares=c[3].number_input("수량",min_value=0.0,step=1.0); price=c[4].number_input("가격",min_value=0.0,step=0.01)
            auto_res = res_t if etf=="TQQQ" else res_s
            stage_default=f"Stage {auto_res['stage']}"
            stage=st.selectbox("당시 단계",["Stage 1","Stage 2","Stage 3","Stage 4","Stage 5","버블보험","월중 비상"], index=["Stage 1","Stage 2","Stage 3","Stage 4","Stage 5","버블보험","월중 비상"].index(stage_default))
            reason=st.text_input("매매 이유", value=auto_res["reason"]); memo=st.text_area("메모")
            submitted=st.form_submit_button("매매 기록 추가", type="primary")
        if submitted:
            if shares <= 0 or price <= 0:
                st.error("수량과 가격은 0보다 커야 합니다.")
            else:
                # Atomic DB function is preferred; fallback keeps compatibility with an existing V2 schema.
                payload={"p_trade_date":str(date),"p_etf":etf,"p_side":side,"p_shares":float(shares),"p_price":float(price),"p_stage":stage,"p_reason":reason,"p_memo":memo}
                ok=False
                try:
                    supabase.rpc("record_trade_and_update_account", payload).execute()
                    ok=True
                except Exception as rpc_error:
                    # Fallback: insert trade, then recalculate account.
                    try:
                        supabase.table("trades").insert({"trade_date":str(date),"etf":etf,"side":side,"shares":float(shares),"price":float(price),"amount":float(shares*price),"stage":stage,"reason":reason,"memo":memo}).execute()
                        acct=load_accounts()
                        row=acct[acct["etf"]==etf]
                        old_shares=float(row["shares"].iloc[0]) if not row.empty else 0.0
                        old_avg=float(row["avg_cost"].iloc[0]) if not row.empty else 0.0
                        old_cash=float(row["cash"].iloc[0]) if not row.empty else 0.0
                        if side=="매수":
                            new_shares=old_shares+shares
                            new_avg=((old_shares*old_avg)+(shares*price))/new_shares if new_shares else 0.0
                            new_cash=old_cash-shares*price
                        else:
                            if shares > old_shares + 1e-9:
                                raise ValueError(f"매도 수량({shares:g})이 현재 보유수량({old_shares:g})보다 많습니다.")
                            new_shares=old_shares-shares
                            new_avg=old_avg if new_shares > 1e-9 else 0.0
                            new_cash=old_cash+shares*price
                        supabase.table("accounts").upsert({"etf":etf,"shares":new_shares,"avg_cost":new_avg,"cash":new_cash,"updated_at":datetime.utcnow().isoformat()}, on_conflict="etf").execute()
                        ok=True
                        st.info("기존 V2 스키마 호환 방식으로 계좌도 갱신했습니다. (권장: V2.2 SQL 함수 설치)")
                    except Exception as fallback_error:
                        st.error(f"매매 저장/계좌 반영 실패: {fallback_error}")
        if ok:
            st.success("매매 기록과 계좌 반영이 완료되었습니다.")
            st.rerun()
        trades=db_select("trades",order=("trade_date",True))
        if not trades.empty:
            st.dataframe(trades.rename(columns={"trade_date":"날짜","etf":"ETF","side":"구분","shares":"수량","price":"가격","amount":"금액","stage":"단계","reason":"매매 이유","memo":"메모"}),use_container_width=True,hide_index=True)
        else: st.info("아직 매매일지가 없습니다.")

# ---------- Performance ----------
elif page == "Performance":
    st.title("📊 Performance")
    if supabase is None:
        st.error("Supabase 연결 후 사용할 수 있습니다.")
    else:
        acct=load_accounts(); perf=[]
        for _,r in acct.iterrows():
            t=r["etf"]; df=market_data.get(t,pd.DataFrame())
            if df.empty: continue
            cur=float(df["Close"].iloc[-1]); avg=float(r["avg_cost"]); shares=float(r["shares"]); pnl=(cur-avg)*shares; cost=avg*shares
            perf.append({"ETF":t,"수익률":pnl/cost*100 if cost else 0,"평가손익":pnl,"평가액":cur*shares})
        if perf:
            p=pd.DataFrame(perf); st.dataframe(p,use_container_width=True,hide_index=True); st.bar_chart(p.set_index("ETF")["수익률"])
        snaps=db_select("snapshots",order=("snapshot_date",True))
        if not snaps.empty:
            st.subheader("📸 전략 스냅샷 기록")
            st.dataframe(snaps[["snapshot_date","etf","stage","stage_name","target_pct","cash_pct","market_dist","etf_dist"]].rename(columns={"snapshot_date":"날짜","etf":"ETF","stage":"단계","stage_name":"단계명","target_pct":"목표비중","cash_pct":"현금비중","market_dist":"시장 이격","etf_dist":"ETF 이격"}),use_container_width=True,hide_index=True)

# ---------- Rules ----------
elif page == "Strategy / Rules":
    st.title("🛡️ Strategy / Rules")
    st.info("현재 V2는 아래 규칙을 코드에 고정하여 전략 자체가 임의로 바뀌지 않도록 했습니다.")
    for ticker in ["TQQQ","SOXL"]:
        r=RULES[ticker]; st.subheader(ticker); data=[]
        for n,name,target,cash in r["stages"]:
            if n==5: cond="하락 후 200일선 재돌파 + 최근 10거래일(약 2주) 연속 200MA 위. 이후 완료 월말 1개월 위면 Stage 1(80%)으로 승격."
            elif n==1: cond=f"{r['market']} > 200일선 (완료 월말 기준 1개월 이상)"
            elif n==2: cond="완료 월말 기준 200일선 첫 이탈"
            elif n==3: cond=r["stage3_text"]
            else: cond=f"{r['market']} 200일선 대비 -25% 이상 급락"
            data.append({"단계":f"Stage {n}","조건":cond,f"{ticker} 비중":f"{target}%","현금":f"{cash}%"})
        st.dataframe(pd.DataFrame(data),use_container_width=True,hide_index=True)
        st.info(f"버블보험: {ticker} 자체 200MA 대비 +{r['bubble']}% 이상 → 주식 비중 60% 수준.\n\n월중 즉시: {r['emergency_text']}")

# ---------- Alerts ----------
elif page == "Alerts":
    st.title("🚨 Alerts")
    alerts=[]
    for t,res in [("TQQQ",res_t),("SOXL",res_s)]:
        if res["bubble"]: alerts.append({"종류":"버블보험","ETF":t,"상태":"HIGH","내용":f"자체 200MA 대비 +{res['etf']['dist']:.2f}% 이상"})
        if res["emergency"]==1: alerts.append({"종류":"월중 비상 1차","ETF":t,"상태":"HIGH","내용":"시장지수 200MA 1차 임계치 도달"})
        if res["emergency"]==2: alerts.append({"종류":"월중 비상 2차","ETF":t,"상태":"CRITICAL","내용":"시장지수 200MA 2차 임계치 도달"})
        if res["stage"] in [2,3,4]: alerts.append({"종류":"단계 변경","ETF":t,"상태":"ACTION","내용":res["action"]})
    if alerts: st.dataframe(pd.DataFrame(alerts),use_container_width=True,hide_index=True)
    else: st.success("현재 활성화된 경보가 없습니다.")

# ---------- Settings ----------
else:
    st.title("⚙️ Settings")
    st.subheader("Supabase 연결 상태")
    if supabase:
        st.markdown('<div class="successbox"><b>연결 성공</b><br>계좌, 매매일지, 전략 스냅샷을 Supabase에 저장할 수 있습니다.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="errorbox"><b>연결 안 됨</b><br>Streamlit Cloud의 Secrets에 SUPABASE_URL과 SUPABASE_SERVICE_ROLE_KEY를 입력하세요.</div>', unsafe_allow_html=True)
    st.subheader("Streamlit Secrets")
    st.code('SUPABASE_URL = "https://YOUR-PROJECT.supabase.co"\nSUPABASE_SERVICE_ROLE_KEY = "YOUR-SERVICE-ROLE-KEY"', language="toml")
    st.warning("SERVICE_ROLE_KEY는 절대 GitHub 코드에 넣지 마세요. Streamlit Secrets에만 저장하세요.")
    st.subheader("저장되는 데이터")
    st.markdown("- **accounts**: TQQQ/SOXL 초기 투자금·보유수량·평단가·현금\n- **trades**: 매수/매도·가격·수량·단계·이유\n- **snapshots**: 날짜별 시장상태·200MA·이격도·Stage·목표비중·판정근거")
    st.subheader("점검 원칙")
    st.write("정상 상태는 2주마다 점검하고, 월중 비상 임계치 도달 시 즉시 대응합니다.")
