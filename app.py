
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from supabase import create_client
except Exception:
    create_client = None

st.set_page_config(page_title="Leveraged Strategy Dashboard V1.1", page_icon="📈", layout="wide")

STRATEGIES = {
    "TQQQ / QQQ": {
        "underlying": "QQQ", "leveraged": "TQQQ",
        "bubble": 30.0, "stage3_gap": -12.0, "stage4_gap": -25.0,
        "weights": {1: 80, 2: 60, 3: 30, 4: 15, "recovery": 60},
        "emergency": [(-15.0, 30), (-25.0, 15)],
    },
    "SOXL / SOXX": {
        "underlying": "SOXX", "leveraged": "SOXL",
        "bubble": 40.0, "stage3_gap": -15.0, "stage4_gap": -25.0,
        "weights": {1: 80, 2: 40, 3: 20, 4: 10, "recovery": 60},
        "emergency": [(-20.0, 20), (-30.0, 10)],
    },
}

# ---------- Supabase ----------
def get_supabase():
    if create_client is None:
        return None
    try:
        url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
        key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))
        if url and key:
            return create_client(url, key)
    except Exception:
        return None
    return None

sb = get_supabase()

def db_select(table, order_col="id"):
    if sb is None:
        return []
    try:
        return sb.table(table).select("*").order(order_col).execute().data or []
    except Exception:
        return []

def db_insert(table, payload):
    if sb is None:
        return False, "Supabase 연결이 없습니다."
    try:
        sb.table(table).insert(payload).execute()
        return True, ""
    except Exception as e:
        return False, str(e)

def db_delete(table, row_id):
    if sb is None:
        return False, "Supabase 연결이 없습니다."
    try:
        sb.table(table).delete().eq("id", row_id).execute()
        return True, ""
    except Exception as e:
        return False, str(e)

# ---------- Market data ----------
@st.cache_data(ttl=900)
def load_prices(tickers):
    if yf is None:
        return {}
    end = datetime.now()
    start = end - timedelta(days=900)
    raw = yf.download(
        tickers, start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False, progress=False, group_by="ticker", threads=True,
    )
    result = {}
    for ticker in tickers:
        try:
            df = raw[ticker].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            df.columns = [str(c).lower() for c in df.columns]
            df = df[["close"]].dropna()
            df["ma200"] = df["close"].rolling(200).mean()
            df["gap"] = (df["close"] / df["ma200"] - 1) * 100
            result[ticker] = df
        except Exception:
            result[ticker] = pd.DataFrame()
    return result

def biweekly_snapshots(df, count=8):
    valid = df.dropna(subset=["ma200"]).copy()
    if valid.empty: return pd.DataFrame()
    target = valid.index[-1]
    rows = []
    for _ in range(count):
        c = valid.loc[valid.index <= target]
        if c.empty: break
        row = c.iloc[-1].copy()
        row.name = c.index[-1]
        rows.append(row)
        target = row.name - pd.Timedelta(days=14)
    return pd.DataFrame(rows[::-1])

def strategy_state(name, data):
    cfg = STRATEGIES[name]
    u, l = data[cfg["underlying"]], data[cfg["leveraged"]]
    if u.empty or l.empty: return {"error": f"{cfg['underlying']} / {cfg['leveraged']} 데이터를 가져오지 못했습니다."}
    snaps = biweekly_snapshots(u, 8)
    if snaps.empty: return {"error": "200일 이동평균 데이터가 부족합니다."}
    gap = float(snaps.iloc[-1]["gap"])
    above = (snaps["gap"] >= 0).tolist()
    below = (snaps["gap"] < 0).tolist()
    ca = 0
    for x in reversed(above):
        if x: ca += 1
        else: break
    cb = 0
    for x in reversed(below):
        if x: cb += 1
        else: break
    prev_gap = float(snaps.iloc[-2]["gap"]) if len(snaps) >= 2 else np.nan
    recross = gap >= 0 and not np.isnan(prev_gap) and prev_gap < 0

    if gap <= cfg["stage4_gap"]:
        stage, title, target = 4, "4단계 · 시발", cfg["weights"][4]
    elif gap <= cfg["stage3_gap"] or cb >= 3:
        stage, title, target = 3, "3단계 · 진성 하락", cfg["weights"][3]
    elif gap < 0:
        stage, title, target = 2, "2단계 · 초기 하락", cfg["weights"][2]
    elif recross:
        stage, title, target = "recovery", "상승 복귀 · 2주차", cfg["weights"]["recovery"]
    elif ca >= 2:
        stage, title, target = 1, "1단계 · 강세장", cfg["weights"][1]
    else:
        stage, title, target = 2, "2단계 · 초기 하락/확인", cfg["weights"][2]

    lr = l.dropna(subset=["ma200"]).iloc[-1]
    lev_gap = float(lr["gap"])
    emergency = next(({"threshold": th, "weight": wt} for th, wt in cfg["emergency"] if gap <= th), None)
    return {
        "cfg": cfg, "snapshots": snaps, "date": snaps.index[-1],
        "price": float(snaps.iloc[-1]["close"]), "ma": float(snaps.iloc[-1]["ma200"]),
        "gap": gap, "stage": stage, "title": title, "target": target,
        "above_count": ca, "below_count": cb, "recross": recross,
        "lev_gap": lev_gap, "emergency": emergency,
    }

# ---------- Trade journal / holdings ----------
def load_trades():
    return pd.DataFrame(db_select("strategy_trades"))

def normalize_trades(df):
    if df.empty: return df
    for c in ["shares", "price", "fx_rate", "fee"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values(["date", "id"])

def calculate_holdings(trades, prices):
    if trades.empty:
        return pd.DataFrame(columns=["ticker","shares","avg_price","cost","current_price","market_value","pnl","return_pct"])
    rows = []
    for ticker, g in trades.groupby(trades["ticker"].str.upper()):
        qty = 0.0
        cost = 0.0
        for _, r in g.sort_values(["date","id"]).iterrows():
            sh = float(r["shares"])
            px = float(r["price"])
            typ = str(r["type"]).upper()
            if typ == "BUY":
                cost += sh * px
                qty += sh
            elif typ == "SELL" and qty > 0:
                avg = cost / qty
                qty -= sh
                cost = max(0, qty * avg)
        if qty <= 1e-9: continue
        current = float(prices.get(ticker, np.nan))
        mv = qty * current if np.isfinite(current) else np.nan
        pnl = mv - cost if np.isfinite(mv) else np.nan
        ret = pnl / cost * 100 if cost else np.nan
        rows.append({
            "ticker": ticker, "shares": qty, "avg_price": cost/qty if qty else 0,
            "cost": cost, "current_price": current, "market_value": mv,
            "pnl": pnl, "return_pct": ret
        })
    return pd.DataFrame(rows)

def fmt_money(x):
    return "-" if pd.isna(x) else f"${x:,.2f}"

# ---------- Sidebar ----------
with st.sidebar:
    st.header("V1.1")
    page = st.radio("메뉴", ["전략 대시보드", "📒 매매일지", "💼 현재 홀딩스"])
    st.caption("전략: 2주 판정 · 월중 비상 대응")
    if st.button("🔄 전체 새로고침"):
        st.cache_data.clear()
        st.rerun()

prices_data = load_prices(["QQQ","TQQQ","SOXX","SOXL"])
prices = {t: float(df["close"].iloc[-1]) for t, df in prices_data.items() if not df.empty}

# ---------- Strategy dashboard ----------
if page == "전략 대시보드":
    st.title("📈 Leveraged Strategy Dashboard V1.1")
    st.caption("매일 계산 · 2주마다 판정 · 급락 시 즉시 비상 대응")

    states = {n: strategy_state(n, prices_data) for n in STRATEGIES}
    for n, s in states.items():
        if "error" in s: st.error(s["error"])

    if all("error" not in s for s in states.values()):
        st.info(f"현재 데이터 기준일: **{max(s['date'] for s in states.values()):%Y-%m-%d}**")
        cols = st.columns(2)
        for col, (name, s) in zip(cols, states.items()):
            with col:
                icon = {1:"🟢",2:"🟡",3:"🟠",4:"🔴","recovery":"🔵"}[s["stage"]]
                st.subheader(name)
                st.markdown(f"### {icon} {s['title']}")
                a,b,c = st.columns(3)
                a.metric(s["cfg"]["underlying"], f"${s['price']:,.2f}")
                b.metric("200MA", f"${s['ma']:,.2f}")
                c.metric("200MA 이격", f"{s['gap']:+.1f}%")
                st.success(f"목표: **{s['cfg']['leveraged']} {s['target']}% / 현금 {100-s['target']}%**")
                if s["emergency"]:
                    e=s["emergency"]
                    st.error(f"🚨 월중 비상: {s['cfg']['underlying']} {s['gap']:+.1f}% → {s['cfg']['leveraged']} **{e['weight']}%** 검토")
                if s["stage"] == 2:
                    st.write(f"200MA 아래 확인: {s['below_count']}회")
                elif s["stage"] == 3:
                    st.write(f"200MA 아래 확인: {s['below_count']}회")
                elif s["stage"] == 1:
                    st.write(f"200MA 위 확인: {s['above_count']}회")
                elif s["stage"] == "recovery":
                    st.write("재돌파 후 2주차 → 비중 회복")
                if s["lev_gap"] >= s["cfg"]["bubble"]:
                    st.warning(f"🔥 버블 보험: {s['cfg']['leveraged']} 200MA 이격 {s['lev_gap']:+.1f}%")

# ---------- Trade journal ----------
elif page == "📒 매매일지":
    st.title("📒 매매일지")
    st.caption("BUY / SELL을 기록하면 현재 홀딩스가 자동으로 계산됩니다.")

    with st.form("trade_form", clear_on_submit=True):
        c1,c2,c3,c4 = st.columns(4)
        date = c1.date_input("거래일", datetime.now().date())
        ticker = c2.text_input("종목", placeholder="TQQQ 또는 SOXL").upper().strip()
        typ = c3.selectbox("구분", ["BUY","SELL"])
        shares = c4.number_input("수량", min_value=0.0001, step=1.0, format="%.4f")
        c5,c6,c7 = st.columns(3)
        price = c5.number_input("주당 가격($)", min_value=0.0, step=0.01, format="%.4f")
        fx = c6.number_input("환율(원/$)", min_value=0.0, value=1400.0, step=1.0)
        fee = c7.number_input("수수료($)", min_value=0.0, step=0.01)
        memo = st.text_input("메모")
        submitted = st.form_submit_button("거래 기록 저장", type="primary")

    if submitted:
        if not ticker or shares <= 0 or price <= 0:
            st.error("종목, 수량, 가격을 입력해 주세요.")
        else:
            ok, msg = db_insert("strategy_trades", {
                "date": str(date), "ticker": ticker, "type": typ,
                "shares": float(shares), "price": float(price),
                "fx_rate": float(fx), "fee": float(fee), "memo": memo
            })
            if ok:
                st.success("거래가 저장되었습니다.")
                st.rerun()
            else:
                st.error(f"저장 실패: {msg}")

    trades = normalize_trades(load_trades())
    if not trades.empty:
        display = trades.copy()
        display["거래일"] = display["date"].dt.strftime("%Y-%m-%d")
        display["금액($)"] = display["shares"] * display["price"]
        display["금액($)"] = display["금액($)"].map(lambda x:f"${x:,.2f}")
        display["환율"] = display["fx_rate"].map(lambda x:f"{x:,.0f}")
        display = display.rename(columns={"ticker":"종목","type":"구분","shares":"수량","price":"가격($)","memo":"메모"})
        st.dataframe(display[["거래일","종목","구분","수량","가격($)","금액($)","환율","메모"]].iloc[::-1], use_container_width=True, hide_index=True)

        st.markdown("#### 거래 삭제")
        options = [f"{r.id} · {r.ticker} · {r.type} · {r.date:%Y-%m-%d} · {r.shares:g}주" for r in trades.itertuples()]
        selected = st.selectbox("삭제할 거래", options)
        if st.button("선택 거래 삭제"):
            row_id = int(selected.split(" · ")[0])
            ok,msg=db_delete("strategy_trades",row_id)
            if ok: st.success("삭제했습니다."); st.rerun()
            else: st.error(msg)
    else:
        st.info("아직 기록된 거래가 없습니다.")

# ---------- Holdings ----------
else:
    st.title("💼 현재 홀딩스")
    st.caption("매매일지의 BUY / SELL을 바탕으로 현재 보유수량과 손익을 자동 계산합니다.")

    trades = normalize_trades(load_trades())
    holdings = calculate_holdings(trades, prices)

    if holdings.empty:
        st.info("매매일지에 BUY 거래를 입력하면 현재 홀딩스가 자동으로 나타납니다.")
    else:
        total = holdings["market_value"].sum()
        total_cost = holdings["cost"].sum()
        total_pnl = holdings["pnl"].sum()
        a,b,c = st.columns(3)
        a.metric("총 평가액", f"${total:,.0f}")
        b.metric("총 매입원가", f"${total_cost:,.0f}")
        c.metric("미실현 손익", f"${total_pnl:+,.0f}")

        h=holdings.copy()
        h["종목"]=h["ticker"]
        h["수량"]=h["shares"].map(lambda x:f"{x:.4f}")
        h["평균매수가"]=h["avg_price"].map(lambda x:f"${x:,.2f}")
        h["현재가"]=h["current_price"].map(lambda x:f"${x:,.2f}" if pd.notna(x) else "-")
        h["평가액"]=h["market_value"].map(lambda x:f"${x:,.0f}" if pd.notna(x) else "-")
        h["손익"]=h["pnl"].map(lambda x:f"${x:+,.0f}" if pd.notna(x) else "-")
        h["수익률"]=h["return_pct"].map(lambda x:f"{x:+.1f}%" if pd.notna(x) else "-")
        st.dataframe(h[["종목","수량","평균매수가","현재가","평가액","손익","수익률"]], use_container_width=True, hide_index=True)

        st.markdown("#### 🎯 전략 목표비중과 비교")
        states = {n: strategy_state(n, prices_data) for n in STRATEGIES}
        target_map = {"TQQQ": None, "SOXL": None}
        for s in states.values():
            if "error" not in s: target_map[s["cfg"]["leveraged"]] = s["target"]
        if total > 0:
            cmp = holdings.copy()
            cmp["실제 비중"] = cmp["market_value"] / total * 100
            cmp["목표 비중"] = cmp["ticker"].map(target_map)
            cmp["차이"] = cmp["목표 비중"] - cmp["실제 비중"]
            cmp["행동"] = cmp["차이"].map(lambda x: "매수 검토" if pd.notna(x) and x > 1 else ("매도 검토" if pd.notna(x) and x < -1 else "유지"))
            cmp["실제 비중"] = cmp["실제 비중"].map(lambda x:f"{x:.1f}%")
            cmp["목표 비중"] = cmp["목표 비중"].map(lambda x:f"{x:.0f}%" if pd.notna(x) else "-")
            cmp["차이"] = cmp["차이"].map(lambda x:f"{x:+.1f}%p" if pd.notna(x) else "-")
            st.dataframe(cmp[["ticker","실제 비중","목표 비중","차이","행동"]].rename(columns={"ticker":"종목"}), use_container_width=True, hide_index=True)
