from datetime import datetime, date
import os
import json
import pandas as pd
import streamlit as st
from supabase import create_client
from pathlib import Path

import numpy as np


# ===== V14 FX-aware transaction ledger =====
def v14_normalize_transactions(tx):
    tx = tx.copy() if tx is not None else pd.DataFrame()
    defaults = {
        "date": "", "ticker": "", "type": "BUY", "shares": 0.0,
        "price": 0.0, "fx_rate": 0.0, "fee_usd": 0.0, "note": ""
    }
    for col, default in defaults.items():
        if col not in tx.columns:
            tx[col] = default
    for col in ["shares", "price", "fx_rate", "fee_usd"]:
        tx[col] = pd.to_numeric(tx[col], errors="coerce").fillna(0.0)
    tx["type"] = tx["type"].astype(str).str.upper().replace({"매수": "BUY", "매도": "SELL"})
    tx["ticker"] = tx["ticker"].astype(str).str.upper()
    tx["date"] = tx["date"].astype(str)
    return tx

def v14_realized_fx_pnl(tx):
    """FIFO: decompose realized KRW P/L into stock-price and FX effects."""
    tx = v14_normalize_transactions(tx)
    if tx.empty:
        return pd.DataFrame()

    result = []
    for ticker in tx["ticker"].dropna().unique():
        lots = []
        rows = tx[tx["ticker"] == ticker].sort_values("date")

        for _, r in rows.iterrows():
            qty = float(r["shares"])
            sell_or_buy_price = float(r["price"])
            fx = float(r["fx_rate"])
            if qty <= 0 or sell_or_buy_price <= 0 or fx <= 0:
                continue

            if r["type"] == "BUY":
                lots.append([qty, sell_or_buy_price, fx, str(r["date"])])
                continue

            if r["type"] != "SELL":
                continue

            remain = qty
            while remain > 1e-10 and lots:
                lot_qty, buy_price, buy_fx, buy_date = lots[0]
                used = min(remain, lot_qty)

                stock_pnl = used * (sell_or_buy_price - buy_price) * buy_fx
                fx_pnl = used * sell_or_buy_price * (fx - buy_fx)
                total_pnl = (
                    used * sell_or_buy_price * fx
                    - used * buy_price * buy_fx
                )

                result.append({
                    "date": str(r["date"]),
                    "ticker": ticker,
                    "shares": used,
                    "buy_price_usd": buy_price,
                    "sell_price_usd": sell_or_buy_price,
                    "buy_fx": buy_fx,
                    "sell_fx": fx,
                    "stock_pnl_krw": stock_pnl,
                    "fx_pnl_krw": fx_pnl,
                    "total_pnl_krw": total_pnl,
                    "stock_return_pct": (sell_or_buy_price / buy_price - 1) * 100,
                    "fx_return_pct": (fx / buy_fx - 1) * 100,
                    "total_return_pct": (
                        (sell_or_buy_price * fx) / (buy_price * buy_fx) - 1
                    ) * 100
                })

                lot_qty -= used
                remain -= used
                if lot_qty <= 1e-10:
                    lots.pop(0)
                else:
                    lots[0][0] = lot_qty

    return pd.DataFrame(result)


# ============================================================
# V15 CLOUD STORAGE
# Uses Supabase when configured; local CSV remains available
# for offline/local use. Supabase credentials belong in
# Streamlit Secrets and must never be committed to GitHub.
# ============================================================
def v15_get_supabase():
    """Create a server-side Supabase client.
    Prefer SUPABASE_SECRET_KEY (never expose it to the browser).
    SUPABASE_KEY remains as a compatibility fallback.
    """
    try:
        url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
        key = st.secrets.get(
            "SUPABASE_SECRET_KEY",
            os.getenv("SUPABASE_SECRET_KEY", "")
        )
        if not key:
            key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))
        if url and key:
            return create_client(url, key)
    except Exception:
        pass
    return None

def v15_empty_transactions():
    return pd.DataFrame({
        "date": pd.Series(dtype="string"),
        "ticker": pd.Series(dtype="string"),
        "type": pd.Series(dtype="string"),
        "shares": pd.Series(dtype="float64"),
        "price": pd.Series(dtype="float64"),
        "fx_rate": pd.Series(dtype="float64"),
        "fee_usd": pd.Series(dtype="float64"),
        "note": pd.Series(dtype="string"),
    })

def v15_load_transactions():
    sb = v15_get_supabase()
    if sb is not None:
        try:
            res = sb.table("transactions").select(
                "date,ticker,type,shares,price,fx_rate,fee_usd,note"
            ).order("date").execute()
            return v14_normalize_transactions(
                pd.DataFrame(res.data or [])
            )
        except Exception:
            st.warning("클라우드 거래내역을 읽지 못했습니다. 로컬 데이터를 사용합니다.")
    path = Path("transactions.csv")
    try:
        return v14_normalize_transactions(
            pd.read_csv(path) if path.exists() else v15_empty_transactions()
        )
    except Exception:
        return v15_empty_transactions()

def v15_save_transactions(tx):
    tx = v14_normalize_transactions(tx).copy()
    tx = tx[[
        "date", "ticker", "type", "shares",
        "price", "fx_rate", "fee_usd", "note"
    ]]
    sb = v15_get_supabase()
    if sb is not None:
        try:
            records = tx.where(pd.notna(tx), None).to_dict(orient="records")
            sb.table("transactions").delete().neq("id", -1).execute()
            if records:
                sb.table("transactions").insert(records).execute()
            return True, "클라우드(Supabase)에 저장했습니다."
        except Exception as e:
            return False, f"클라우드 저장 실패: {e}"
    tx.to_csv("transactions.csv", index=False)
    return True, "PC의 transactions.csv에 저장했습니다."

def v15_cloud_status():
    return v15_get_supabase() is not None


# ============================================================
# V16 Portfolio Intelligence Engine
# ============================================================
V16_STOP_LOSS_PCT = -32.0
V16_TRAILING_STOP_PCT = -10.0
V16_PARTIAL_SELL_PCT = 30.0

V16_CATEGORY_MAP = {
    "NVDA":"AI / GPU","AVGO":"AI / Networking","MSFT":"AI / Software",
    "GOOG":"AI / Software","AMZN":"AI / Cloud","META":"AI / Software",
    "MU":"Memory","WDC":"Memory","SNDK":"Memory",
    "ANET":"Networking","MRVL":"Networking",
    "VRT":"Power / Data Center","NVT":"Power / Data Center",
    "COHR":"Optical","GLW":"Optical",
    "KLAC":"Semiconductor Equipment","LRCX":"Semiconductor Equipment",
    "AMAT":"Semiconductor Equipment","ASML":"Semiconductor Equipment",
    "PANW":"Cybersecurity","CRWD":"Cybersecurity",
    "AAPL":"Mega Cap Tech","QQQ":"ETF","TQQQ":"ETF",
    "UPRO":"ETF","SOXL":"ETF","QLD":"ETF"
}

def v16_num(v, default=0.0):
    try:
        x = float(v)
        return default if pd.isna(x) else x
    except Exception:
        return default

def v16_category(ticker, fallback="Other"):
    return V16_CATEGORY_MAP.get(str(ticker).upper(), fallback)

def v16_tx_normalize(tx):
    if tx is None or tx.empty:
        return pd.DataFrame(columns=["date","ticker","type","shares","price","fx_rate"])
    x = tx.copy()
    ren = {}
    for c in x.columns:
        lc = str(c).lower()
        if lc in ("종목","ticker","symbol"): ren[c]="ticker"
        elif lc in ("구분","type","side"): ren[c]="type"
        elif lc in ("수량","shares","qty"): ren[c]="shares"
        elif lc in ("주가","price","매수가","매도가"): ren[c]="price"
        elif lc in ("환율","fx_rate","fx"): ren[c]="fx_rate"
        elif lc in ("날짜","date"): ren[c]="date"
    x=x.rename(columns=ren)
    for c in ["date","ticker","type","shares","price","fx_rate"]:
        if c not in x.columns: x[c] = "" if c in ["date","ticker","type"] else 0
    x["ticker"]=x["ticker"].astype(str).str.upper().str.strip()
    x["type"]=x["type"].astype(str).str.upper().str.strip()
    x["shares"]=pd.to_numeric(x["shares"],errors="coerce").fillna(0)
    x["price"]=pd.to_numeric(x["price"],errors="coerce").fillna(0)
    x["fx_rate"]=pd.to_numeric(x["fx_rate"],errors="coerce").fillna(0)
    x["date"]=pd.to_datetime(x["date"],errors="coerce")
    return x.sort_values(["date"]).reset_index(drop=True)

def v16_build_holdings(tx, fallback=None):
    x=v16_tx_normalize(tx)
    if x.empty:
        if fallback is not None and not fallback.empty:
            f=fallback.copy()
            f["ticker"]=f["ticker"].astype(str).str.upper()
            for c in ["shares","avg_price"]:
                if c not in f: f[c]=0
            f["category"]=f["ticker"].map(v16_category)
            f["avg_fx"]=0.0
            return f[["ticker","shares","avg_price","avg_fx","category"]]
        return pd.DataFrame(columns=["ticker","shares","avg_price","avg_fx","category"])
    out=[]
    for ticker, g in x.groupby("ticker", sort=True):
        lots=[]
        for _,r in g.iterrows():
            q=v16_num(r.shares); p=v16_num(r.price); fx=v16_num(r.fx_rate)
            if q<=0 or p<=0: continue
            if r["type"]=="BUY":
                lots.append([q,p,fx])
            elif r["type"]=="SELL":
                rem=q
                while rem>1e-9 and lots:
                    take=min(rem,lots[0][0])
                    lots[0][0]-=take; rem-=take
                    if lots[0][0]<=1e-9: lots.pop(0)
        shares=sum(z[0] for z in lots)
        if shares>1e-9:
            avg=sum(z[0]*z[1] for z in lots)/shares
            avg_fx=sum(z[0]*z[2] for z in lots)/shares if any(z[2] for z in lots) else 0
            out.append({"ticker":ticker,"shares":shares,"avg_price":avg,
                        "avg_fx":avg_fx,"category":v16_category(ticker)})
    return pd.DataFrame(out)

def v16_market_data(tickers, fallback):
    data={}
    static={}
    if fallback is not None and not fallback.empty:
        for _,r in fallback.iterrows():
            static[str(r.get("ticker","")).upper()]=v16_num(r.get("current_price",r.get("price",r.get("avg_price",0))))
    try:
        import yfinance as yf
        for t in tickers:
            try:
                h=yf.Ticker(t).history(period="2y", auto_adjust=True)
                if h is None or h.empty: continue
                close=h["Close"].dropna()
                if close.empty: continue
                data[t]={
                    "price":float(close.iloc[-1]),
                    "ma200":float(close.tail(200).mean()),
                    "high_52w":float(close.tail(252).max()),
                    "series":close
                }
            except Exception: pass
    except Exception: pass
    for t in tickers:
        if t not in data:
            p=static.get(t,0)
            data[t]={"price":p,"ma200":p,"high_52w":p,"series":pd.Series(dtype=float)}
    return data

def v16_alerts(holdings, prices, previous_highs=None):
    previous_highs = previous_highs or {}
    rows=[]
    for _,r in holdings.iterrows():
        t=str(r.ticker).upper()
        md=prices.get(t,{})
        price=v16_num(md.get("price"))
        avg=v16_num(r.avg_price)
        shares=v16_num(r.shares)

        # Best available high. V16.1 prefers a stored high, then the
        # available market history high. This is exposed explicitly.
        history_high=v16_num(md.get("high_52w"))
        stored_high=v16_num(previous_highs.get(t))
        high=max(price, stored_high, history_high)

        ret=(price/avg-1)*100 if avg else 0.0
        ma200=v16_num(md.get("ma200"))
        ma_gap=(price/ma200-1)*100 if ma200 else 0.0
        from_high=(price/high-1)*100 if high else 0.0

        if ret <= V16_STOP_LOSS_PCT:
            status="STOP LOSS"
            action="손절 검토"
        elif from_high <= V16_TRAILING_STOP_PCT:
            status="TRAILING STOP"
            action=f"{shares*V16_PARTIAL_SELL_PCT/100:.2f}주 (30%) 부분매도 검토"
        elif ma200 and price < ma200:
            status="200MA BELOW"
            action="200일선 아래 — 추세 주의"
        else:
            status="NORMAL"
            action="정상"

        rows.append({
            "ticker":t, "shares":shares, "price":price, "avg_price":avg,
            "return_pct":ret, "ma200":ma200, "ma_gap_pct":ma_gap,
            "high":high, "from_high_pct":from_high,
            "partial_shares":shares*V16_PARTIAL_SELL_PCT/100,
            "status":status, "action":action
        })
    return pd.DataFrame(rows)

def v16_sell_review(realized, current_prices):
    """Summarize realized SELL matches and calculate post-sale return."""
    if realized is None or realized.empty:
        return pd.DataFrame()
    x = realized.copy()
    x["shares"] = pd.to_numeric(x["shares"], errors="coerce").fillna(0)
    x["sell_price_usd"] = pd.to_numeric(x["sell_price_usd"], errors="coerce").fillna(0)
    x["total_pnl_krw"] = pd.to_numeric(x["total_pnl_krw"], errors="coerce").fillna(0)
    rows=[]
    for (d,t), g in x.groupby(["date","ticker"], sort=True):
        qty=g["shares"].sum()
        sell_price=(g["shares"]*g["sell_price_usd"]).sum()/qty if qty else 0
        pnl=g["total_pnl_krw"].sum()
        current=v16_num(current_prices.get(str(t).upper(),{}).get("price"))
        post=(current/sell_price-1)*100 if sell_price and current else np.nan
        rows.append({
            "매도일":str(d), "종목":str(t).upper(), "수량":qty,
            "매도가":sell_price, "실현손익_원":pnl,
            "승패":"승" if pnl > 0 else ("패" if pnl < 0 else "보합"),
            "매도후수익률":post, "현재가":current
        })
    return pd.DataFrame(rows)

def v16_daily_direction(prices, tickers, days=10):
    rows=[]
    for t in tickers:
        md=prices.get(str(t).upper(),{})
        s=md.get("series")
        if s is None or len(s)<2: continue
        s=pd.Series(s).dropna().tail(days+1)
        if len(s)<2: continue
        prev=None
        for dt, close in s.iloc[1:].items():
            prior=float(s.loc[prev]) if prev is not None else None
            chg=(float(close)/prior-1)*100 if prior else np.nan
            direction="🟢 상승" if chg>0 else ("🔴 하락" if chg<0 else "⚪ 보합")
            rows.append({"종목":str(t).upper(),"날짜":pd.to_datetime(dt).strftime("%Y-%m-%d"),"종가":float(close),"일일변동":chg,"방향":direction})
            prev=dt
    return pd.DataFrame(rows)

def v16_direction_summary(prices, tickers, window=5):
    rows=[]
    for t in tickers:
        s=prices.get(str(t).upper(),{}).get("series")
        if s is None or len(s)<window+1: continue
        s=pd.Series(s).dropna().tail(window+1)
        rets=s.pct_change().dropna()*100
        up=int((rets>0).sum()); down=int((rets<0).sum())
        cum=(float(s.iloc[-1])/float(s.iloc[0])-1)*100
        if up>=4 and cum>0: trend="🟢 상승"
        elif down>=4 and cum<0: trend="🔴 하락"
        else: trend="🟡 횡보/혼조"
        rows.append({"종목":str(t).upper(),"최근5일":cum,"상승일":up,"하락일":down,"단기추세":trend})
    return pd.DataFrame(rows)

def v16_upsert_daily_direction(daily_df):
    if daily_df is None or daily_df.empty: return False
    try:
        sb=v15_get_supabase()
        if sb is None: return False
        records=[]
        for _,r in daily_df.iterrows():
            records.append({"date":str(r["날짜"]),"ticker":str(r["종목"]),"close_price":float(r["종가"]),"daily_return_pct":float(r["일일변동"])})
        sb.table("daily_directions").upsert(records, on_conflict="date,ticker").execute()
        return True
    except Exception:
        return False

def v16_upsert_snapshot(value, pnl, count):
    try:
        sb=v15_get_supabase()
        if sb is None or value<=0: return False
        d=datetime.now().date().isoformat()
        sb.table("portfolio_values").upsert({
            "date":d,"value_usd":float(value),
            "holdings_count":int(count),"total_pnl_usd":float(pnl)
        }, on_conflict="date").execute()
        return True
    except Exception:
        return False

def v16_load_history():
    try:
        sb=v15_get_supabase()
        if sb is None: return pd.DataFrame()
        res=sb.table("portfolio_values").select("*").order("date").execute()
        return pd.DataFrame(res.data or [])
    except Exception:
        return pd.DataFrame()


st.set_page_config(page_title="My Portfolio", page_icon="∞", layout="wide", initial_sidebar_state="expanded")

# ---------- CSS ----------
st.markdown("""
<style>
.stApp { background:#0b0d0f; color:#f4f6f8; }
.block-container { padding:1rem 2rem 3rem; max-width:1500px; }
[data-testid="stSidebar"] { background:#080a0c; border-right:1px solid #1b2025; }
[data-testid="stSidebar"] * { color:#dfe5eb !important; }
h1,h2,h3,h4 { color:#f4f6f8 !important; }
p, label { color:#8f98a3 !important; }
.card { background:#121518; border:1px solid #242a30; border-radius:16px; padding:20px; height:100%; }
.big { font-size:2.15rem; font-weight:750; letter-spacing:-.04em; color:#fff; }
.metric-label { color:#8f98a3; font-size:.8rem; margin-bottom:5px; }
.metric-value { color:#fff; font-size:1.65rem; font-weight:700; }
.small { font-size:.78rem; color:#8f98a3; }
.positive { color:#23d98b; font-weight:650; }
.negative { color:#ff5b63; font-weight:650; }
.badge { display:inline-block; padding:4px 8px; border-radius:7px; background:#18251f; color:#25db8c; font-size:.75rem; }
.ticker { background:#111519; border:1px solid #252b31; border-radius:999px; padding:8px 13px; text-align:center; font-size:.78rem; }
.ticker b { color:#f5f7f9; margin-right:6px; }
.section-title { font-size:1.15rem; font-weight:700; color:#fff; margin-bottom:12px; }
.asset-row { display:grid; grid-template-columns:1.25fr .65fr .75fr .85fr .8fr; gap:10px; padding:12px 0; border-bottom:1px solid #20262b; align-items:center; }
.asset-head { color:#78828d; font-size:.72rem; }
.asset-name { color:#fff; font-weight:650; }
.asset-sub { color:#737e89; font-size:.7rem; }
.info { background:#10181d; border:1px solid #1e3038; border-radius:12px; padding:14px; color:#aab5bf; }
</style>
""", unsafe_allow_html=True)

# ---------- Data ----------
df = pd.read_csv("portfolio.csv")
prices = {"NVDA":180,"AVGO":380,"MSFT":520,"GOOG":205,"MU":165,"ANET":145,"VRT":205,"COHR":180,"KLAC":1450,"LRCX":145}
daily = {"NVDA":2.5,"AVGO":1.8,"MSFT":1.3,"GOOG":0.8,"MU":3.1,"ANET":2.0,"VRT":1.5,"COHR":4.2,"KLAC":1.1,"LRCX":1.9}
df["current_price"] = df.ticker.map(prices).fillna(df.avg_price)
df["daily_pct"] = df.ticker.map(daily).fillna(0)
df["market_value"] = df.shares * df.current_price
df["cost_basis"] = df.shares * df.avg_price
df["pnl"] = df.market_value - df.cost_basis
df["return_pct"] = np.where(df.cost_basis != 0, df.pnl / df.cost_basis * 100, 0)
total_value = df.market_value.sum()
total_cost = df.cost_basis.sum()
total_pnl = df.pnl.sum()
total_return = total_pnl / total_cost * 100 if total_cost else 0
df["weight"] = df.market_value / total_value * 100
df["contribution"] = df.pnl / total_cost * 100


# ---------- V16 live portfolio calculation ----------
try:
    v16_tx_live = v15_load_transactions()
except Exception:
    v16_tx_live = pd.DataFrame()

try:
    v16_fallback = pd.read_csv("portfolio.csv")
except Exception:
    v16_fallback = pd.DataFrame(columns=["ticker","shares","avg_price"])

v16_hold = v16_build_holdings(v16_tx_live, v16_fallback)
v16_prices = v16_market_data(v16_hold["ticker"].tolist() if not v16_hold.empty else [], v16_fallback)

if not v16_hold.empty:
    v16_hold["current_price"] = v16_hold["ticker"].map(lambda t:v16_num(v16_prices.get(t,{}).get("price")))
    v16_hold["market_value"] = v16_hold["shares"] * v16_hold["current_price"]
    v16_hold["cost_basis"] = v16_hold["shares"] * v16_hold["avg_price"]
    v16_hold["pnl"] = v16_hold["market_value"] - v16_hold["cost_basis"]
    v16_hold["return_pct"] = (v16_hold["market_value"]/v16_hold["cost_basis"]-1)*100
    v16_total_value=float(v16_hold["market_value"].sum())
    v16_total_pnl=float(v16_hold["pnl"].sum())
    v16_hold["weight"]=v16_hold["market_value"]/v16_total_value*100 if v16_total_value else 0
else:
    v16_total_value=0.0
    v16_total_pnl=0.0

v16_alert_df=v16_alerts(v16_hold, v16_prices)
# Merge risk/alert analytics back into Holdings so the table shows the
# same numbers used by Alert Center.
if not v16_hold.empty and not v16_alert_df.empty:
    _v16_alert_cols = [
        "ticker","ma200","ma_gap_pct","high","from_high_pct",
        "status","action","partial_shares"
    ]
    v16_hold = v16_hold.merge(
        v16_alert_df[_v16_alert_cols],
        on="ticker", how="left", suffixes=("","_alert")
    )

v16_stop_df=v16_alert_df[v16_alert_df["status"]=="STOP LOSS"].copy()
v16_trail_df=v16_alert_df[v16_alert_df["status"]=="TRAILING STOP"].copy()

# Save a daily portfolio snapshot. It is harmless if the table does not exist yet.
v16_upsert_snapshot(v16_total_value, v16_total_pnl, len(v16_hold))
v16_history=v16_load_history()
v16_daily_df = v16_daily_direction(v16_prices, v16_hold["ticker"].tolist() if not v16_hold.empty else [], days=10)
v16_direction_summary_df = v16_direction_summary(v16_prices, v16_hold["ticker"].tolist() if not v16_hold.empty else [], window=5)
v16_upsert_daily_direction(v16_daily_df)


# ---------- Sidebar navigation ----------
with st.sidebar:
    st.markdown("## ∞")
    st.markdown("### Portfolio")
    page = st.radio(
        "Navigation",
        ["▦  Overview", "▣  Holdings", "▥  Performance", "◫  Allocation", "◴  History", "⚙  Settings"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown('<div class="small">V3 · Navigation enabled</div>', unsafe_allow_html=True)

def header():
    st.markdown("### 🏠  /  " + page.replace("▦  ","").replace("▣  ","").replace("▥  ","").replace("◫  ","").replace("◴  ","").replace("⚙  ",""))
    tickers = ""
    for _, r in df.sort_values("weight", ascending=False).head(7).iterrows():
        cls = "positive" if r.daily_pct >= 0 else "negative"
        tickers += f'<div class="ticker"><b>{r.ticker}</b> ${r.current_price:,.2f} <span class="{cls}">{r.daily_pct:+.1f}%</span></div>'
    st.markdown(f'<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:8px;margin:10px 0 18px">{tickers}</div>', unsafe_allow_html=True)

header()

# ---------- Overview ----------


# ---------- V16 Alert Center ----------

if "page" in locals() and str(page).startswith(("▦","Overview","overview")):
    if not v16_stop_df.empty or not v16_trail_df.empty:
        st.markdown("## 🚨 ALERT CENTER")
        if not v16_stop_df.empty:
            st.error("🔴 STOP LOSS — 수익률 -32% 이하. 손절을 검토하세요.")
            for _,a in v16_stop_df.iterrows():
                st.markdown(
                    f"**{a.ticker}** · 현재 수익률 **{a.return_pct:+.1f}%** · "
                    f"손절 기준 {V16_STOP_LOSS_PCT:.0f}% · **손절 검토**"
                )
        if not v16_trail_df.empty:
            st.warning("🟠 TRAILING STOP — 전고점 대비 -10% 이하. 30% 부분매도를 검토하세요.")
            for _,a in v16_trail_df.iterrows():
                st.markdown(
                    f"**{a.ticker}** · 전고점 ${a.high:,.2f} → 현재 ${a.price:,.2f} "
                    f"({a.from_high_pct:+.1f}%) · **{a.partial_shares:.2f}주 (30%) 부분매도 검토**"
                )
    else:
        st.success("🟢 No Critical Alerts — 현재 손절/트레일링 스톱 조건에 해당하는 종목이 없습니다.")

if page.startswith("▦"):
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Portfolio value", f"${total_value:,.0f}")
    c2.metric("Invested", f"${total_cost:,.0f}")
    c3.metric("Total P&L", f"${total_pnl:,.0f}", f"{total_return:+.2f}%")
    c4.metric("Holdings", f"{len(df)}")

    st.markdown("---")
    left,right = st.columns([1.15,1])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Value trend & impact</div>', unsafe_allow_html=True)
        x=np.arange(12); y=total_value*(.91+.01*x+.018*np.sin(x*1.1))
        st.line_chart(pd.DataFrame({"Portfolio":y}, index=[f"{i+1}M" for i in x]), height=230)
        st.markdown('<div class="small">V3 demo history. Real daily history will be connected later.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Risk snapshot</div>', unsafe_allow_html=True)
        st.metric("Top 5 concentration", f"{df.nlargest(5,'market_value').weight.sum():.1f}%")
        ai = df[df.category.isin(["AI","Memory","Optical","Networking","Power","Semiconductor Equipment"])].market_value.sum()/total_value*100
        st.metric("AI / infrastructure exposure", f"{ai:.1f}%")
        st.metric("Largest position", f"{df.loc[df.market_value.idxmax(),'ticker']} · {df.weight.max():.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    a,b = st.columns([1,1])
    with a:
        st.markdown('<div class="card"><div class="section-title">Theme allocation</div>', unsafe_allow_html=True)
        theme = df.groupby("category").market_value.sum()/total_value*100
        st.bar_chart(theme, height=240)
        st.markdown('</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><div class="section-title">Asset performance</div>', unsafe_allow_html=True)
        for _,r in df.sort_values("weight",ascending=False).iterrows():
            cls="positive" if r.return_pct>=0 else "negative"
            st.markdown(f'<div class="asset-row"><div><div class="asset-name">{r.ticker}</div><div class="asset-sub">{r.category}</div></div><div>{r.weight:.1f}%</div><div class="{cls}">{r.return_pct:+.1f}%</div><div class="{cls}">{r.contribution:+.2f}%</div><div>${r.market_value:,.0f}</div></div>',unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Holdings ----------
elif page.startswith("▣"):
    st.markdown(
        '<div class="info">거래일지의 BUY/SELL 기록을 기준으로 보유수량과 평균매수가를 계산하고, '
        '현재가·200일선·전고점 기준의 리스크 상태를 함께 표시합니다.</div>',
        unsafe_allow_html=True
    )
    if v16_hold.empty:
        st.info("보유 종목이 없습니다. 거래일지에 BUY를 입력해 주세요.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("보유 종목", f"{len(v16_hold)}개")
        total_mv = float(v16_hold["market_value"].sum())
        top5 = float(v16_hold.nlargest(5,"market_value")["market_value"].sum()) / total_mv * 100 if total_mv else 0
        c2.metric("상위 5 비중", f"{top5:.1f}%")
        c3.metric("총 평가손익", f"${v16_hold['pnl'].sum():,.0f}")
        c4.metric("Risk Signals", f"{int((v16_alert_df['status'] != 'NORMAL').sum())}개")

        st.markdown("#### Risk & Action Monitor")
        view = v16_hold.sort_values("weight", ascending=False).copy()
        view["종목"] = view["ticker"]
        view["테마"] = view["category"]
        view["수량"] = view["shares"].map(lambda x:f"{x:.4g}")
        view["매입단가"] = view["avg_price"].map(lambda x:f"${x:,.2f}")
        view["현재가"] = view["current_price"].map(lambda x:f"${x:,.2f}")
        view["평가액"] = view["market_value"].map(lambda x:f"${x:,.0f}")
        view["비중"] = view["weight"].map(lambda x:f"{x:.1f}%")
        view["수익률"] = view["return_pct"].map(lambda x:f"{x:+.1f}%")
        view["200일선 이격"] = view["ma_gap_pct"].map(lambda x:f"{x:+.1f}%")
        view["전고점"] = view["high"].map(lambda x:f"${x:,.2f}")
        view["전고점 대비"] = view["from_high_pct"].map(lambda x:f"{x:+.1f}%")
        view["상태"] = view["status"].map({
            "STOP LOSS":"🔴 STOP LOSS",
            "TRAILING STOP":"🟠 TRAILING STOP",
            "200MA BELOW":"🟡 200MA BELOW",
            "NORMAL":"🟢 NORMAL"
        }).fillna("⚪ UNKNOWN")
        view["액션"] = view["action"].fillna("정상")
        if not v16_direction_summary_df.empty:
            _dir = v16_direction_summary_df.set_index("종목")
            view["최근5일"] = view["ticker"].map(_dir["최근5일"]).map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "-")
            view["단기추세"] = view["ticker"].map(_dir["단기추세"]).fillna("⚪ 데이터 부족")
        else:
            view["최근5일"] = "-"
            view["단기추세"] = "⚪ 데이터 부족"

        st.dataframe(
            view[["종목","테마","수량","매입단가","현재가","평가액","비중",
                  "수익률","200일선 이격","전고점","전고점 대비","최근5일","단기추세","상태","액션"]],
            use_container_width=True, hide_index=True
        )
        st.caption("200일선 이격 = 현재가와 200일 이동평균선의 거리 · 전고점 대비 = 현재가와 가용 최고가의 차이 · 최근5일/단기추세는 일봉 종가 기준")

        st.markdown("#### 📈 Daily Direction — 최근 10거래일")
        if not v16_daily_df.empty:
            dd = v16_daily_df.copy()
            dd["종가"] = dd["종가"].map(lambda x:f"${x:,.2f}")
            dd["일일변동"] = dd["일일변동"].map(lambda x:f"{x:+.2f}%")
            st.dataframe(dd[["종목","날짜","종가","일일변동","방향"]], use_container_width=True, hide_index=True)
        else:
            st.info("일봉 데이터를 불러오지 못했습니다. yfinance 연결을 확인하세요.")

        active = view[view["status"] != "NORMAL"]
        if not active.empty:
            st.markdown("#### ⚠️ Active Signals")
            for _, row in active.iterrows():
                if row["status"] == "STOP LOSS":
                    st.error(f"{row['종목']} · {row['수익률']} · **STOP LOSS → 손절 검토**")
                elif row["status"] == "TRAILING STOP":
                    qty = v16_num(row["shares"]) * V16_PARTIAL_SELL_PCT / 100
                    st.warning(f"{row['종목']} · 전고점 대비 {row['전고점 대비']} · **{qty:.2f}주 (30%) 부분매도 검토**")
                elif row["status"] == "200MA BELOW":
                    st.warning(f"{row['종목']} · 200일선 이격 {row['200일선 이격']} · **200일선 아래 — 추세 주의**")

# ---------- Placeholder pages ----------
elif page.startswith("▥"):
    st.info("Performance 페이지 — 다음 버전에서 기간별 수익률, QQQ/SPY 비교, 종목별 수익 기여도를 구현합니다.")
elif page.startswith("◫"):
    st.info("Allocation 페이지 — AI/메모리/광통신/네트워크/전력/장비 등 실제 테마 노출도를 종목별로 분해해 보여줄 예정입니다.")
elif page.startswith("◴"):
    st.info("History 페이지 — 거래내역과 포트폴리오 가치 변화를 저장하고 과거 시점의 포트폴리오를 볼 수 있게 만들 예정입니다.")
else:
    st.info("Settings 페이지 — 목표비중, 테마 분류, 시장 전략 설정 등을 관리하게 만들 예정입니다.")



# ===== V14: FX-aware Transaction Journal =====
st.markdown("### 💱 Transaction Journal — V14")
st.caption("매수·매도 당시 환율을 저장하여 주가손익과 환차익/환차손을 분리합니다.")

v14_tx_path = Path("transactions.csv")
v14_tx = v15_load_transactions()
v14_tx = v14_normalize_transactions(v14_tx)
v14_cols = [
    "date", "ticker", "type", "shares",
    "price", "fx_rate", "fee_usd", "note"
]

v14_editor_df = v14_tx[v14_cols].copy() if not v14_tx.empty else v15_empty_transactions()
v14_edited = st.data_editor(
    v14_editor_df,
    num_rows="dynamic",
    use_container_width=True,
    key="v14_transaction_editor",
    column_config={
        "date": st.column_config.TextColumn("날짜"),
        "ticker": st.column_config.TextColumn("종목"),
        "type": st.column_config.SelectboxColumn(
            "구분", options=["BUY", "SELL"]
        ),
        "shares": st.column_config.NumberColumn(
            "수량", min_value=0.0, step=1.0
        ),
        "price": st.column_config.NumberColumn(
            "주가(USD)", min_value=0.0, step=0.01, format="$%.2f"
        ),
        "fx_rate": st.column_config.NumberColumn(
            "거래환율(원/$)", min_value=0.0, step=1.0, format="%.0f"
        ),
        "fee_usd": st.column_config.NumberColumn(
            "수수료(USD)", min_value=0.0, step=0.01
        ),
        "note": st.column_config.TextColumn("메모")
    }
)

if st.button("💾 거래내역 저장", key="v15_save_transactions"):
    ok, msg = v15_save_transactions(v14_edited)
    if ok:
        st.success(msg)
        st.rerun()
    else:
        st.error(msg)

v14_realized = v14_realized_fx_pnl(v14_edited)

if not v14_realized.empty:
    st.markdown("#### 실현손익 — 주가손익 vs 환차손익")

    v14_stock_total = v14_realized["stock_pnl_krw"].sum()
    v14_fx_total = v14_realized["fx_pnl_krw"].sum()
    v14_total = v14_realized["total_pnl_krw"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("주가손익", f"₩{v14_stock_total:,.0f}")
    c2.metric("환차익 / 환차손", f"₩{v14_fx_total:,.0f}")
    c3.metric("총 실현손익", f"₩{v14_total:,.0f}")

    v14_show = v14_realized[[
        "date", "ticker", "shares",
        "buy_price_usd", "sell_price_usd",
        "buy_fx", "sell_fx",
        "stock_pnl_krw", "fx_pnl_krw", "total_pnl_krw",
        "stock_return_pct", "fx_return_pct", "total_return_pct"
    ]].copy()

    v14_show.columns = [
        "매도일", "종목", "수량",
        "매수가($)", "매도가($)",
        "매수환율", "매도환율",
        "주가손익(원)", "환차손익(원)", "총손익(원)",
        "주가수익률(%)", "환율수익률(%)", "총수익률(%)"
    ]

    st.dataframe(
        v14_show,
        use_container_width=True,
        hide_index=True
    )

    # V16.2 — Sell Review
    st.markdown("#### 🏆 매도 성과 — 승/패 & 매도 후 수익률")
    v16_sell = v16_sell_review(v14_realized, v16_prices)
    if not v16_sell.empty:
        wins = int((v16_sell["승패"] == "승").sum())
        losses = int((v16_sell["승패"] == "패").sum())
        decided = wins + losses
        win_rate = wins / decided * 100 if decided else 0
        avg_post = pd.to_numeric(v16_sell["매도후수익률"], errors="coerce").mean()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("승", f"{wins}회")
        c2.metric("패", f"{losses}회")
        c3.metric("승률", f"{win_rate:.1f}%")
        c4.metric("매도 후 평균", f"{avg_post:+.1f}%" if pd.notna(avg_post) else "-")

        show = v16_sell.copy()
        show["매도가"] = show["매도가"].map(lambda x:f"${x:,.2f}")
        show["실현손익"] = show["실현손익_원"].map(lambda x:f"₩{x:,.0f}")
        show["매도후수익률"] = show["매도후수익률"].map(lambda x:f"{x:+.1f}%" if pd.notna(x) else "-")
        show["현재가"] = show["현재가"].map(lambda x:f"${x:,.2f}" if x else "-")
        st.dataframe(show[["매도일","종목","수량","매도가","실현손익","승패","현재가","매도후수익률"]].rename(columns={"매도일":"매도일","매도후수익률":"매도 후 수익률"}), use_container_width=True, hide_index=True)
        st.caption("승/패는 매도 시점의 실현손익 기준입니다. '매도 후 수익률'은 매도가 대비 현재가의 변화입니다.")

