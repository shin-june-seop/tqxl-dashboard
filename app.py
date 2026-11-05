from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Any

import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

try:
    from supabase import create_client
except ImportError:  # pragma: no cover
    create_client = None

APP_VERSION = "1.0.1"
CURRENCY = "USD"

# Strategy: each ETF has its own budget and one-time staged purchase plan.
# SPYM is the S&P 500 sleeve; its official S&P 500 benchmark period begins 2020-01-24.
STRATEGIES: dict[str, dict[str, Any]] = {
    "SPYM": {
        "name": "SPDR Portfolio S&P 500 ETF",
        "allocation": 0.60,
        "drawdown_reference": "2020-01-24",
        "stages": [
            {"key": "D1", "label": "1차", "drawdown": -8.0, "weight": 0.20},
            {"key": "D2", "label": "2차", "drawdown": -12.0, "weight": 0.15},
            {"key": "D3", "label": "3차", "drawdown": -16.0, "weight": 0.15},
            {"key": "D4", "label": "4차", "drawdown": -20.0, "weight": 0.15},
            {"key": "D5", "label": "5차", "drawdown": -24.0, "weight": 0.15},
            {"key": "RECOVERY", "label": "회복 확인", "drawdown": None, "weight": 0.20},
        ],
    },
    "SOXX": {
        "name": "iShares Semiconductor ETF",
        "allocation": 0.40,
        "drawdown_reference": "2001-07-13",
        "stages": [
            {"key": "D1", "label": "1차", "drawdown": -25.0, "weight": 0.20},
            {"key": "D2", "label": "2차", "drawdown": -30.0, "weight": 0.15},
            {"key": "D3", "label": "3차", "drawdown": -35.0, "weight": 0.15},
            {"key": "D4", "label": "4차", "drawdown": -40.0, "weight": 0.15},
            {"key": "D5", "label": "5차", "drawdown": -45.0, "weight": 0.15},
            {"key": "RECOVERY", "label": "회복 확인", "drawdown": None, "weight": 0.20},
        ],
    },
}

st.set_page_config(
    page_title="SPYM + SOXX 분할매수 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1450px; padding-top: 1.2rem; padding-bottom: 2.5rem;}
    .hero {padding: 1.15rem 1.35rem; border-radius: 16px; background: linear-gradient(125deg,#101827,#1d2c43); border:1px solid #34435a; margin-bottom: 1rem;}
    .hero h1 {margin:0; font-size:1.85rem; color:#f7f9fc;}
    .hero p {margin:.35rem 0 0; color:#c7d3e4;}
    .ticker-card {padding:1rem 1.1rem; border-radius:14px; border:1px solid #344154; background:rgba(30,43,62,.35);}
    .ticker-name {font-size:.88rem; color:#aab8ca;}
    .ticker-price {font-size:1.8rem; font-weight:750; margin:.15rem 0;}
    .muted {color:#8795a8; font-size:.82rem;}
    .section-note {color:#8795a8; font-size:.82rem;}
    div[data-testid="stMetric"] {background:rgba(128,128,128,.06); padding:.65rem .8rem; border-radius:12px; border:1px solid rgba(128,128,128,.15);}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- Data -----------------------------------------

def _normalize_history(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).title() for c in out.columns]
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    out.index = idx.normalize()
    if "Close" not in out.columns:
        return pd.DataFrame()
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]:
        if col not in out.columns:
            out[col] = 0.0 if col in ["Dividends", "Stock Splits", "Volume"] else pd.NA
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out["Adj Close"] = pd.to_numeric(out["Adj Close"], errors="coerce").fillna(out["Close"])
    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").fillna(0)
    out = out.dropna(subset=["Close"])
    return out[~out.index.duplicated(keep="last")].sort_index()


def _ensure_split_adjusted(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, list[str]]:
    """Normalize pre-split OHLC prices to the latest share unit when the feed leaves a split gap.

    Yahoo data can temporarily mix pre-split historical closes with post-split current prices.
    We detect this at each known/action-reported split by comparing the last pre-split close
    with the first ex-split close. If the ratio is approximately the split ratio, historical
    prices are divided and volume is multiplied. If the feed is already continuous, no change.
    """
    if df.empty:
        return df, []
    out = df.copy()
    if ticker != "SOXX":
        return out, []

    events: dict[pd.Timestamp, float] = {
        pd.Timestamp("2024-03-07"): 3.0,
        pd.Timestamp("2026-11-05"): 3.0,
    }
    if "Stock Splits" in out.columns:
        reported = pd.to_numeric(out["Stock Splits"], errors="coerce").fillna(0.0)
        for d, ratio in reported[reported > 0].items():
            events[pd.Timestamp(d).normalize()] = float(ratio)

    notes: list[str] = []
    price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in out.columns]
    for split_date, ratio in sorted(events.items()):
        if ratio <= 1 or out.index.min() >= split_date or out.index.max() < split_date:
            continue
        post_dates = out.index[out.index >= split_date]
        if len(post_dates) == 0:
            continue
        first_post = post_dates[0]
        prior_dates = out.index[out.index < first_post]
        if len(prior_dates) == 0:
            continue
        prev_close = float(out.loc[prior_dates[-1], "Close"])
        post_close = float(out.loc[first_post, "Close"])
        if prev_close <= 0 or post_close <= 0:
            continue
        observed_ratio = prev_close / post_close
        # A split gap is present only when the market-price ratio is close to the
        # corporate-action ratio. Ordinary price fluctuations must not trigger adjustment.
        if ratio * 0.75 <= observed_ratio <= ratio * 1.25:
            mask = out.index < first_post
            out.loc[mask, price_cols] = out.loc[mask, price_cols].apply(pd.to_numeric, errors="coerce") / ratio
            if "Volume" in out.columns:
                out.loc[mask, "Volume"] = pd.to_numeric(out.loc[mask, "Volume"], errors="coerce") * ratio
            notes.append(f"{split_date.date()} {ratio:g}:1 split 전 가격을 분할 후 기준으로 환산")
        elif 0.75 <= observed_ratio <= 1.25:
            notes.append(f"{split_date.date()} {ratio:g}:1 split은 데이터에 이미 반영됨")
        else:
            notes.append(
                f"{split_date.date()} split 조정 상태 불확실(전후 가격비 {observed_ratio:.2f}, 예상 {ratio:g}); 자동 조정하지 않음"
            )
    return out, notes


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(ticker: str) -> tuple[pd.DataFrame, str]:
    """Fetch split-adjusted daily OHLC data. SPYM falls back to legacy SPLG history."""
    if yf is None:
        return pd.DataFrame(), "yfinance 패키지가 설치되지 않았습니다."

    def get_one(symbol: str) -> pd.DataFrame:
        try:
            raw = yf.Ticker(symbol).history(
                period="max", interval="1d", auto_adjust=False, actions=True
            )
            return _normalize_history(raw)
        except Exception:
            return pd.DataFrame()

    if ticker == "SPYM":
        current = get_one("SPYM")
        legacy = get_one("SPLG")
        cutoff = pd.Timestamp("2025-10-31")
        # The ETF's ticker changed from SPLG to SPYM on 2025-10-31.
        # Prefer current SPYM rows from the change date onward and legacy SPLG rows before it.
        if not legacy.empty:
            legacy = legacy.loc[legacy.index < cutoff]
        if not current.empty:
            current = current.loc[current.index >= cutoff] if current.index.min() < cutoff else current
        frames = [x for x in [legacy, current] if not x.empty]
        if frames:
            merged = pd.concat(frames).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            source = "Yahoo Finance · SPYM + SPLG 과거 데이터 연결"
            return merged, source
        return pd.DataFrame(), "SPYM 및 과거 티커 SPLG 데이터를 가져오지 못했습니다."

    data = get_one(ticker)
    data, split_notes = _ensure_split_adjusted(data, ticker)
    source = "Yahoo Finance · split 단위 검증 후 가격 정규화"
    if split_notes:
        source += " · " + " / ".join(split_notes)
    return data, source


# ----------------------------- Supabase -------------------------------------

@st.cache_resource(show_spinner=False)
def get_supabase():
    if create_client is None:
        return None, "supabase 패키지가 설치되지 않았습니다."
    try:
        url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
        key = st.secrets.get(
            "SUPABASE_SERVICE_ROLE_KEY",
            os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
        )
        if not url or not key:
            return None, "Supabase Secrets가 설정되지 않았습니다."
        return create_client(url, key), None
    except Exception as exc:
        return None, f"Supabase 연결 실패: {exc}"


supabase, supabase_error = get_supabase()


def db_get_config() -> dict[str, Any]:
    default = {"id": "default", "total_budget": 10000.0}
    if supabase is None:
        return st.session_state.get("local_config", default)
    try:
        rows = supabase.table("split_strategy_config").select("*").eq("id", "default").limit(1).execute().data or []
        if rows:
            return rows[0]
        supabase.table("split_strategy_config").upsert(default, on_conflict="id").execute()
        return default
    except Exception as exc:
        st.warning(f"설정 테이블을 읽지 못했습니다. Supabase SQL을 먼저 실행해 주세요. ({exc})")
        return default


def db_save_config(total_budget: float) -> bool:
    payload = {"id": "default", "total_budget": float(total_budget), "updated_at": datetime.utcnow().isoformat()}
    if supabase is None:
        st.session_state["local_config"] = payload
        return True
    try:
        supabase.table("split_strategy_config").upsert(payload, on_conflict="id").execute()
        return True
    except Exception as exc:
        st.error(f"투자 예산 저장 실패: {exc}")
        return False


def db_get_state(ticker: str) -> dict[str, Any] | None:
    if supabase is None:
        return st.session_state.setdefault("local_states", {}).get(ticker)
    try:
        rows = supabase.table("split_strategy_state").select("*").eq("ticker", ticker).limit(1).execute().data or []
        return rows[0] if rows else None
    except Exception as exc:
        st.warning(f"{ticker} 전략 상태를 읽지 못했습니다. SQL 설치 여부를 확인해 주세요. ({exc})")
        return None


def db_save_state(ticker: str, state: dict[str, Any]) -> bool:
    payload = {
        "ticker": ticker,
        "tracking_start_date": state["tracking_start_date"],
        "triggered_stages": state.get("triggered_stages", []),
        "stage_trigger_dates": state.get("stage_trigger_dates", {}),
        "recovery_triggered": bool(state.get("recovery_triggered", False)),
        "recovery_date": state.get("recovery_date"),
        "updated_at": datetime.utcnow().isoformat(),
    }
    if supabase is None:
        st.session_state.setdefault("local_states", {})[ticker] = payload
        return True
    try:
        supabase.table("split_strategy_state").upsert(payload, on_conflict="ticker").execute()
        return True
    except Exception as exc:
        st.error(f"{ticker} 전략 단계 저장 실패: {exc}")
        return False


def db_get_trades() -> pd.DataFrame:
    if supabase is None:
        rows = st.session_state.setdefault("local_trades", [])
        return pd.DataFrame(rows)
    try:
        rows = supabase.table("split_strategy_trades").select("*").order("trade_date", desc=True).order("id", desc=True).execute().data or []
        return pd.DataFrame(rows)
    except Exception as exc:
        st.warning(f"매수 기록을 불러오지 못했습니다. Supabase SQL을 먼저 실행해 주세요. ({exc})")
        return pd.DataFrame()


def db_add_trade(payload: dict[str, Any]) -> tuple[bool, str]:
    if supabase is None:
        rows = st.session_state.setdefault("local_trades", [])
        payload = dict(payload)
        payload["id"] = len(rows) + 1
        rows.append(payload)
        return True, "현재 세션에 매수 기록을 저장했습니다. Supabase 미연결 상태에서는 재접속 후 기록이 사라질 수 있습니다."
    try:
        supabase.table("split_strategy_trades").insert(payload).execute()
        return True, "매수 기록을 저장했습니다."
    except Exception as exc:
        return False, f"매수 기록 저장 실패: {exc}"


def db_delete_trade(trade_id: int) -> tuple[bool, str]:
    if supabase is None:
        rows = st.session_state.setdefault("local_trades", [])
        st.session_state["local_trades"] = [r for r in rows if int(r.get("id", -1)) != int(trade_id)]
        return True, "매수 기록을 삭제했습니다."
    try:
        supabase.table("split_strategy_trades").delete().eq("id", int(trade_id)).execute()
        return True, "매수 기록을 삭제했습니다."
    except Exception as exc:
        return False, f"삭제 실패: {exc}"


# ----------------------------- Strategy -------------------------------------

def parse_date(value: Any, fallback: date) -> date:
    try:
        return pd.Timestamp(value).date()
    except Exception:
        return fallback


def evaluate_state(ticker: str, df: pd.DataFrame, saved: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Persist one-time drawdown triggers and first 20DMA reclaim after a trigger."""
    rule = STRATEGIES[ticker]
    if df.empty:
        return {}, {"error": "가격 데이터 없음"}

    ref_date = pd.Timestamp(rule["drawdown_reference"])
    calc = df.loc[df.index >= ref_date].copy()
    if calc.empty:
        return {}, {"error": f"{rule['drawdown_reference']} 이후 데이터 없음"}

    close = calc["Close"].astype(float)
    high_watermark = close.cummax()
    drawdown = (close / high_watermark - 1.0) * 100.0
    sma20 = close.rolling(20, min_periods=20).mean()

    latest_date = close.index[-1].date()
    if saved:
        tracking_start = parse_date(saved.get("tracking_start_date"), latest_date)
    else:
        # On first launch, only recognize drawdown levels already reached as of the latest close.
        tracking_start = latest_date

    # Rebuild stage dates from the split-normalized price history instead of blindly trusting
    # persisted trigger flags. This also repairs stages falsely triggered by a split-price gap.
    monitor = drawdown.loc[drawdown.index.date >= tracking_start]
    triggered: set[str] = set()
    stage_dates: dict[str, str] = {}
    for stage in rule["stages"]:
        if stage["drawdown"] is None:
            continue
        key = stage["key"]
        hits = monitor[monitor <= float(stage["drawdown"]) + 1e-9]
        if not hits.empty:
            triggered.add(key)
            stage_dates[key] = hits.index[0].date().isoformat()

    recovery_triggered = False
    recovery_date = None
    first_drawdown_dates = [pd.Timestamp(v) for k, v in stage_dates.items() if k != "RECOVERY"]
    if first_drawdown_dates:
        first_drawdown_date = min(first_drawdown_dates)
        prev_close = close.shift(1)
        prev_sma = sma20.shift(1)
        reclaim = (close > sma20) & (prev_close <= prev_sma)
        reclaim = reclaim & (pd.Series(calc.index, index=calc.index) > first_drawdown_date)
        recovery_hits = calc.index[reclaim.fillna(False)]
        if len(recovery_hits):
            recovery_triggered = True
            recovery_date = recovery_hits[0].date().isoformat()
            triggered.add("RECOVERY")
            stage_dates["RECOVERY"] = recovery_date

    new_state = {
        "tracking_start_date": tracking_start.isoformat(),
        "triggered_stages": [s["key"] for s in rule["stages"] if s["key"] in triggered],
        "stage_trigger_dates": stage_dates,
        "recovery_triggered": recovery_triggered,
        "recovery_date": recovery_date,
    }

    latest_price = float(close.iloc[-1])
    latest_high = float(high_watermark.iloc[-1])
    latest_dd = float(drawdown.iloc[-1])
    latest_sma20 = float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else None
    above_20 = latest_sma20 is not None and latest_price > latest_sma20
    previous_above_20 = None
    if len(close) >= 2 and pd.notna(sma20.iloc[-2]):
        previous_above_20 = bool(close.iloc[-2] > sma20.iloc[-2])

    details = {
        "price": latest_price,
        "date": latest_date,
        "high_watermark": latest_high,
        "drawdown": latest_dd,
        "sma20": latest_sma20,
        "above_20": above_20,
        "previous_above_20": previous_above_20,
        "triggered": new_state["triggered_stages"],
        "stage_dates": stage_dates,
        "tracking_start": tracking_start,
        "max_drawdown_since_tracking": float(monitor.min()) if not monitor.empty else latest_dd,
        "data_start": calc.index[0].date(),
    }
    return new_state, details


def trade_totals(trades: pd.DataFrame, ticker: str) -> dict[str, Any]:
    if trades.empty:
        return {"shares": 0.0, "cost": 0.0, "by_stage": {}}
    t = trades[trades["ticker"].astype(str) == ticker].copy()
    if t.empty:
        return {"shares": 0.0, "cost": 0.0, "by_stage": {}}
    t["amount"] = pd.to_numeric(t["amount"], errors="coerce").fillna(0.0)
    t["shares"] = pd.to_numeric(t["shares"], errors="coerce").fillna(0.0)
    grouped = t.groupby("stage_key", dropna=False).agg(amount=("amount", "sum"), shares=("shares", "sum"))
    by_stage = {str(k): float(v) for k, v in grouped["amount"].to_dict().items()}
    return {"shares": float(t["shares"].sum()), "cost": float(t["amount"].sum()), "by_stage": by_stage}


def money(value: float) -> str:
    return f"${value:,.2f}"


def pct(value: float | None) -> str:
    return "확인 불가" if value is None else f"{value:+.2f}%"


# ----------------------------- Header / Sidebar -----------------------------

with st.sidebar:
    st.title("📈 SPYM + SOXX")
    st.caption(f"Split Buy Dashboard · v{APP_VERSION}")
    st.markdown("**전략 목표 비중**")
    st.markdown("- SPYM 60%\n- SOXX 40%")
    if st.button("🔄 시장 데이터 새로고침", use_container_width=True):
        fetch_history.clear()
        st.rerun()
    st.divider()
    if supabase is not None:
        st.success("Supabase 연결됨 · 영구 저장")
    else:
        st.warning("Supabase 미연결 · 기록은 현재 세션에만 저장")
        st.caption("영구 저장을 위해 프로젝트의 supabase_schema.sql을 Supabase SQL Editor에서 실행하고 Secrets를 설정하세요.")

st.markdown(
    '<div class="hero"><h1>SPYM + SOXX 분할매수 대시보드</h1>'
    '<p>SPYM 60% · SOXX 40% | 낙폭 단계별 매수 · 20일선 회복 매수 · 장기 보유</p></div>',
    unsafe_allow_html=True,
)

with st.spinner("SPYM과 SOXX의 최신 일별 가격 데이터를 확인하고 있습니다..."):
    market_data: dict[str, pd.DataFrame] = {}
    data_sources: dict[str, str] = {}
    for ticker in STRATEGIES:
        market_data[ticker], data_sources[ticker] = fetch_history(ticker)

with st.expander("데이터 출처 및 분할 조정 점검", expanded=False):
    for ticker in STRATEGIES:
        st.caption(f"{ticker}: {data_sources[ticker]}")

missing = [t for t, df in market_data.items() if df.empty]
if missing:
    st.error(f"가격 데이터를 가져오지 못했습니다: {', '.join(missing)}. 잠시 후 새로고침하거나 Yahoo Finance 데이터 상태를 확인하세요.")
    st.stop()

# ----------------------------- Budget ---------------------------------------
config = db_get_config()
current_budget = float(config.get("total_budget", 10000.0) or 10000.0)
with st.expander("⚙️ 투자 예산 설정", expanded=False):
    st.caption("총 투자예산을 입력하면 SPYM 60%, SOXX 40%로 각 ETF의 분할매수 금액이 자동 계산됩니다. 통화는 USD입니다.")
    b1, b2 = st.columns([2, 1])
    proposed_budget = b1.number_input("전체 투자예산 (USD)", min_value=0.0, value=current_budget, step=500.0, format="%.2f")
    if b2.button("예산 저장", type="primary", use_container_width=True):
        if db_save_config(proposed_budget):
            st.success("투자예산을 저장했습니다.")
            st.rerun()

budget = float(proposed_budget if "proposed_budget" in locals() else current_budget)

# ----------------------------- State update ---------------------------------
states: dict[str, dict[str, Any]] = {}
metrics: dict[str, dict[str, Any]] = {}
for ticker, rule in STRATEGIES.items():
    saved_state = db_get_state(ticker)
    new_state, detail = evaluate_state(ticker, market_data[ticker], saved_state)
    if "error" in detail:
        st.error(f"{ticker}: {detail['error']}")
        st.stop()
    db_save_state(ticker, new_state)
    states[ticker] = new_state
    metrics[ticker] = detail

trades = db_get_trades()

# ----------------------------- Summary --------------------------------------
allocations = {ticker: budget * rule["allocation"] for ticker, rule in STRATEGIES.items()}
totals = {ticker: trade_totals(trades, ticker) for ticker in STRATEGIES}
account_rows: list[dict[str, Any]] = []
portfolio_market_value = 0.0
portfolio_cost = 0.0
portfolio_cash = 0.0
portfolio_assets = 0.0

for ticker, rule in STRATEGIES.items():
    current_price = metrics[ticker]["price"]
    t = totals[ticker]
    market_value = t["shares"] * current_price
    cost = t["cost"]
    cash_remaining = allocations[ticker] - cost
    account_value = market_value + cash_remaining
    pnl = market_value - cost
    return_pct = (pnl / cost * 100.0) if cost > 0 else None
    account_rows.append({
        "ticker": ticker,
        "shares": t["shares"],
        "avg_cost": cost / t["shares"] if t["shares"] > 0 else None,
        "market_value": market_value,
        "cost": cost,
        "cash": cash_remaining,
        "account_value": account_value,
        "pnl": pnl,
        "return_pct": return_pct,
    })
    portfolio_market_value += market_value
    portfolio_cost += cost
    portfolio_cash += cash_remaining
    portfolio_assets += account_value

portfolio_pnl = portfolio_market_value - portfolio_cost
portfolio_return = portfolio_pnl / portfolio_cost * 100.0 if portfolio_cost > 0 else None
portfolio_total_return = (portfolio_assets / budget - 1.0) * 100.0 if budget > 0 else None

# Amounts activated by triggers but not yet purchased.
for ticker, rule in STRATEGIES.items():
    t = totals[ticker]
    for stage in rule["stages"]:
        stage["planned_amount"] = allocations[ticker] * stage["weight"]
        stage["purchased_amount"] = float(t.get("by_stage", {}).get(stage["key"], 0.0))
        stage["remaining_amount"] = max(0.0, stage["planned_amount"] - stage["purchased_amount"])
        stage["triggered"] = stage["key"] in states[ticker].get("triggered_stages", [])
        stage["due_amount"] = stage["remaining_amount"] if stage["triggered"] else 0.0

# KPI cards
k1, k2, k3, k4 = st.columns(4)
k1.metric("전체 투자예산", money(budget))
k2.metric("현재 ETF 평가액", money(portfolio_market_value))
k3.metric("미투입 현금", money(portfolio_cash))
k4.metric("전체 계좌 평가액", money(portfolio_assets), delta=f"총 예산 대비 {pct(portfolio_total_return)}" if portfolio_total_return is not None else None)

if supabase is None:
    st.warning("현재는 임시 세션 저장 모드입니다. 브라우저 세션이 끝나면 매수 기록과 단계 진행 내역이 유지되지 않을 수 있습니다. 정식 사용 전 Supabase SQL 설정을 완료하세요.")

# ----------------------------- ETF cards ------------------------------------
card_cols = st.columns(2)
for col, ticker in zip(card_cols, STRATEGIES):
    rule = STRATEGIES[ticker]
    m = metrics[ticker]
    state = states[ticker]
    triggered_count = sum(1 for s in rule["stages"][:-1] if s["key"] in state["triggered_stages"])
    due = sum(float(s["due_amount"]) for s in rule["stages"])
    price_vs_20 = ((m["price"] / m["sma20"] - 1.0) * 100.0) if m["sma20"] else None
    col.markdown(f"""
    <div class="ticker-card">
      <div class="ticker-name">{ticker} · {rule['name']}</div>
      <div class="ticker-price">{money(m['price'])}</div>
      <div class="muted">기준일 {m['date']} · 전고점 {money(m['high_watermark'])}</div>
      <hr style="border-color:#35445a; margin:.65rem 0;">
      <div><b>고점 대비 낙폭</b> &nbsp; <span style="font-size:1.25rem;font-weight:700;">{m['drawdown']:.2f}%</span></div>
      <div class="muted">현재 단계: 낙폭 매수 {triggered_count}/5 · 20일선 대비 {pct(price_vs_20)}</div>
      <div style="margin-top:.55rem;"><b>실행 대기 금액</b> &nbsp; <span style="font-size:1.3rem;font-weight:750;">{money(due)}</span></div>
      <div class="muted">배정 예산 {money(allocations[ticker])} · 누적 매수 {money(totals[ticker]['cost'])}</div>
    </div>
    """, unsafe_allow_html=True)

st.caption("낙폭은 split-adjusted 종가 기준의 누적 최고 종가 대비 하락률입니다. 낙폭 신호는 종가 기준으로 발생하며, 대시보드는 주문을 실행하지 않습니다.")

# ----------------------------- Stage tables ---------------------------------
st.subheader("🎯 분할매수 단계 및 투입 금액")
st.caption("각 매수 단계는 ETF별 배정 예산의 비율입니다. 조건이 한 번 발생하면 완료 처리되어 이후 가격이 반등해도 단계가 사라지지 않습니다. 각 단계는 한 번만 사용합니다.")
for ticker, rule in STRATEGIES.items():
    st.markdown(f"#### {ticker} 매수 계획")
    rows = []
    for stage in rule["stages"]:
        if stage["key"] == "RECOVERY":
            condition = "하락 단계 발생 후 종가가 20일선을 아래에서 위로 상향 돌파"
        else:
            condition = f"전고점 대비 {stage['drawdown']:.0f}% 이하"
        if stage["triggered"]:
            status = "매수 완료" if stage["remaining_amount"] <= 0.01 else ("부분 매수" if stage["purchased_amount"] > 0 else "매수 대기")
        else:
            status = "신호 대기"
        rows.append({
            "단계": stage["label"],
            "조건": condition,
            "배정 비중": f"{stage['weight']*100:.0f}%",
            "예정 금액": money(stage["planned_amount"]),
            "실제 매수": money(stage["purchased_amount"]),
            "남은 금액": money(stage["remaining_amount"]),
            "상태": status,
            "신호일": states[ticker].get("stage_trigger_dates", {}).get(stage["key"], "-"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ----------------------------- Account / weights ----------------------------
st.subheader("💼 실제 보유·수익률·비중")
portfolio_table = []
for row in account_rows:
    ticker = row["ticker"]
    allocation_pct = STRATEGIES[ticker]["allocation"] * 100.0
    actual_weight = row["market_value"] / portfolio_assets * 100.0 if portfolio_assets > 0 else 0.0
    avg_cost_text = money(row["avg_cost"]) if row["avg_cost"] is not None else "-"
    return_text = f"{row['return_pct']:+.2f}%" if row["return_pct"] is not None else "-"
    portfolio_table.append({
        "ETF": ticker,
        "목표 비중": f"{allocation_pct:.0f}%",
        "실제 평가 비중": f"{actual_weight:.1f}%",
        "보유 수량": f"{row['shares']:.6f}",
        "평균 매수가": avg_cost_text,
        "현재 평가액": money(row["market_value"]),
        "매수 원금": money(row["cost"]),
        "가격 손익": money(row["pnl"]),
        "가격 수익률": return_text,
        "배정 현금 잔액": money(row["cash"]),
    })
portfolio_table.append({
    "ETF": "현금",
    "목표 비중": "-",
    "실제 평가 비중": f"{portfolio_cash / portfolio_assets * 100.0:.1f}%" if portfolio_assets > 0 else "-",
    "보유 수량": "-", "평균 매수가": "-", "현재 평가액": money(portfolio_cash),
    "매수 원금": "-", "가격 손익": "-", "가격 수익률": "-", "배정 현금 잔액": money(portfolio_cash),
})
st.dataframe(pd.DataFrame(portfolio_table), use_container_width=True, hide_index=True)
st.caption("실제 평가 비중은 SPYM 평가액·SOXX 평가액·미투입 현금을 합한 계좌 평가액 기준입니다. 가격 손익과 수익률은 매수 기록의 체결가 대비 현재가 기준이며 배당/분배금은 포함하지 않습니다.")

pnl1, pnl2, pnl3 = st.columns(3)
pnl1.metric("누적 매수 원금", money(portfolio_cost))
pnl2.metric("미실현 가격 손익", money(portfolio_pnl), delta=f"{portfolio_return:+.2f}%" if portfolio_return is not None else None)
pnl3.metric("예산 대비 계좌 변화", money(portfolio_pnl), delta=f"{portfolio_total_return:+.2f}%" if portfolio_total_return is not None else None)

# ----------------------------- Trade entry ----------------------------------
st.subheader("📝 실제 매수 기록")
st.caption("매수 체결 후 금액과 체결가를 입력하면 수량·평단·수익률·비중에 반영됩니다. 실제 주문은 증권사에서 직접 실행하세요.")
trade_choices = []
for ticker, rule in STRATEGIES.items():
    for stage in rule["stages"]:
        trade_choices.append(f"{ticker} · {stage['label']} ({stage['key']})")

with st.form("add_trade_form", clear_on_submit=True):
    c1, c2, c3, c4 = st.columns([1, 1.5, 1, 1])
    trade_ticker = c1.selectbox("ETF", list(STRATEGIES.keys()))
    stage_map = {f"{t} · {s['label']} ({s['key']})": (t, s["key"]) for t, r in STRATEGIES.items() for s in r["stages"]}
    trade_stage_label = c2.selectbox("매수 단계", [x for x in trade_choices if x.startswith(trade_ticker + " · ")])
    trade_date = c3.date_input("체결일", value=metrics[trade_ticker]["date"])
    trade_price = c4.number_input("체결가 ($)", min_value=0.0, value=float(metrics[trade_ticker]["price"]), step=0.01, format="%.4f")
    amount_col, note_col = st.columns([1, 2])
    stage_key = stage_map[trade_stage_label][1]
    stage_def = next(s for s in STRATEGIES[trade_ticker]["stages"] if s["key"] == stage_key)
    suggested_amount = max(0.0, stage_def["planned_amount"] - stage_def["purchased_amount"])
    trade_amount = amount_col.number_input("매수 금액 ($)", min_value=0.0, value=float(round(suggested_amount, 2)), step=50.0, format="%.2f")
    trade_note = note_col.text_input("메모 (선택)", placeholder="예: 다음 거래일 시가 매수")
    submit_trade = st.form_submit_button("매수 기록 저장", type="primary", use_container_width=True)

if submit_trade:
    if trade_price <= 0 or trade_amount <= 0:
        st.error("체결가와 매수 금액은 0보다 커야 합니다.")
    elif stage_key not in states[trade_ticker].get("triggered_stages", []):
        st.error("아직 매수 신호가 발생하지 않은 단계입니다. 조건 충족 후 기록해 주세요.")
    elif suggested_amount <= 0.01:
        st.error("이 단계의 예정 금액은 이미 모두 매수 처리되었습니다.")
    elif trade_amount > suggested_amount + 0.01:
        st.error(f"이 단계의 남은 예정 금액은 {money(suggested_amount)}입니다. 초과 입력을 확인해 주세요.")
    else:
        shares = trade_amount / trade_price
        payload = {
            "ticker": trade_ticker,
            "stage_key": stage_key,
            "trade_date": trade_date.isoformat(),
            "price": float(trade_price),
            "amount": float(trade_amount),
            "shares": float(shares),
            "note": trade_note.strip(),
        }
        ok, message = db_add_trade(payload)
        (st.success if ok else st.error)(f"{message} · 기록 수량 {shares:.6f}주")
        if ok:
            st.rerun()

# ----------------------------- Trade history -------------------------------
if not trades.empty:
    st.subheader("📚 매수 기록")
    display_trades = trades.copy()
    display_trades["trade_date"] = display_trades["trade_date"].astype(str)
    display_trades["수량"] = pd.to_numeric(display_trades["shares"], errors="coerce").fillna(0).map(lambda x: f"{x:.6f}")
    display_trades["체결가"] = pd.to_numeric(display_trades["price"], errors="coerce").fillna(0).map(money)
    display_trades["금액"] = pd.to_numeric(display_trades["amount"], errors="coerce").fillna(0).map(money)
    display_trades = display_trades.rename(columns={"trade_date": "체결일", "ticker": "ETF", "stage_key": "단계", "note": "메모"})
    cols = [c for c in ["id", "체결일", "ETF", "단계", "수량", "체결가", "금액", "메모"] if c in display_trades.columns]
    st.dataframe(display_trades[cols], use_container_width=True, hide_index=True)
    with st.expander("잘못 입력한 최근 매수 기록 삭제"):
        if "id" in trades.columns:
            ids = trades.sort_values("id", ascending=False)["id"].tolist()[:20]
            id_to_delete = st.selectbox("삭제할 기록 ID", ids)
            if st.button("선택한 기록 삭제", type="secondary"):
                ok, message = db_delete_trade(int(id_to_delete))
                (st.success if ok else st.error)(message)
                if ok:
                    st.rerun()

# ----------------------------- Footer / methodology -------------------------
st.divider()
with st.expander("📌 전략 규칙 및 계산 기준", expanded=False):
    st.markdown(
        """
        **목표 자산배분**
        - SPYM 60%, SOXX 40%. 각 ETF의 분할매수 예산은 전체 예산에 이 비중을 곱해 계산합니다.

        **SPYM 분할매수**
        - -8%, -12%, -16%, -20%, -24%에서 각각 SPYM 배정 예산의 20%, 15%, 15%, 15%, 15%를 투입합니다.
        - 나머지 20%는 하락 단계가 한 번 이상 발생한 뒤 20일 이동평균선을 종가 기준 아래에서 위로 상향 돌파할 때 투입 대상으로 표시합니다.

        **SOXX 분할매수**
        - -25%, -30%, -35%, -40%, -45%에서 각각 SOXX 배정 예산의 20%, 15%, 15%, 15%, 15%를 투입합니다.
        - 나머지 20%는 같은 20일선 상향 돌파 조건에서 투입 대상으로 표시합니다.

        **신호 및 보유 원칙**
        - 낙폭은 split-adjusted 종가의 누적 최고 종가 대비 하락률로 계산합니다. 당일 종가로 조건을 판정하고 실제 매수는 사용자가 증권사에서 실행합니다.
        - 최초 실행 시점에는 현재 가격이 이미 임계값 아래라면 해당 단계들을 통과한 것으로 표시합니다. 이후 발생한 단계 신호는 Supabase에 저장되며, 가격이 반등해도 사라지지 않습니다.
        - 각 단계는 ETF별로 한 번만 사용합니다. 새 고점을 기록해도 사용한 단계 예산은 초기화하지 않습니다. 이는 매도 없이 예산을 한 번 배치하는 장기 보유형 규칙입니다.
        - 회복 매수는 하락 단계 이후 종가가 20일선을 아래에서 위로 교차한 경우에만 신호로 처리합니다. 첫 설치 이전에 발생한 과거 회복 교차는 소급하지 않습니다.

        **SPYM 데이터 처리**
        - SPYM은 2025-10-31에 SPLG에서 티커가 변경됐으므로 Yahoo Finance의 과거 SPLG 자료를 연결해 가격 이력을 보완합니다.
        - SPYM의 현재 S&P 500 추종 기준은 State Street가 밝힌 2020-01-24 이후로 제한해 고점 및 낙폭을 계산합니다.

        **수익률 및 현금**
        - 보유 수량은 매수 금액 ÷ 체결가로 계산합니다. 평가손익은 현재 split-adjusted 종가와 매수 기록으로 계산하며 배당·분배금, 세금, 환전 비용 및 수수료는 제외합니다.
        - 미투입 현금은 ETF별 배정 예산에서 기록된 매수 금액을 뺀 값으로 계산합니다. 현금 이자는 반영하지 않습니다.
        - 이 대시보드는 신호·금액·기록 관리용이며 자동 주문을 실행하지 않습니다.
        """
    )

st.caption(f"가격 데이터: {data_sources['SPYM']} / {data_sources['SOXX']} · 마지막 데이터: SPYM {metrics['SPYM']['date']} / SOXX {metrics['SOXX']['date']} · {CURRENCY}")
st.caption("SPYM 60 : SOXX 40 · No 200-day moving-average sell rule · Buy-and-hold after each recorded purchase")
