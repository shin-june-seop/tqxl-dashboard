
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Leveraged Strategy Dashboard V1",
    page_icon="📈",
    layout="wide",
)

# -----------------------------
# Config
# -----------------------------
STRATEGIES = {
    "TQQQ / QQQ": {
        "underlying": "QQQ",
        "leveraged": "TQQQ",
        "bubble": 30.0,
        "stage3_gap": -12.0,
        "stage3_gap_high": -15.0,
        "stage4_gap": -25.0,
        "weights": {1: 80, 2: 60, 3: 30, 4: 15, "recovery": 60},
        "emergency": [(-15.0, 30), (-25.0, 15)],
    },
    "SOXL / SOXX": {
        "underlying": "SOXX",
        "leveraged": "SOXL",
        "bubble": 40.0,
        "stage3_gap": -15.0,
        "stage3_gap_high": -15.0,
        "stage4_gap": -25.0,
        "weights": {1: 80, 2: 40, 3: 20, 4: 10, "recovery": 60},
        "emergency": [(-20.0, 20), (-30.0, 10)],
    },
}

# -----------------------------
# Data
# -----------------------------
@st.cache_data(ttl=900)
def load_data(tickers):
    import yfinance as yf

    end = datetime.now()
    start = end - timedelta(days=900)

    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    result = {}
    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw[ticker].copy()
            else:
                df = raw.copy()

            df = df.rename(columns=str.lower)
            df = df[["close"]].dropna().copy()
            df["ma200"] = df["close"].rolling(200).mean()
            df["gap"] = (df["close"] / df["ma200"] - 1) * 100
            result[ticker] = df
        except Exception:
            result[ticker] = pd.DataFrame()

    return result


def biweekly_snapshots(df, count=8):
    """Build checkpoints about 14 calendar days apart, using the nearest prior trading day."""
    if df.empty:
        return pd.DataFrame()

    valid = df.dropna(subset=["ma200"]).copy()
    if valid.empty:
        return valid

    latest = valid.index[-1]
    rows = []
    target = latest

    for _ in range(count):
        candidates = valid.loc[valid.index <= target]
        if candidates.empty:
            break
        row = candidates.iloc[-1].copy()
        row.name = candidates.index[-1]
        rows.append(row)
        target = row.name - pd.Timedelta(days=14)

    return pd.DataFrame(rows[::-1])


def count_consecutive(values, condition):
    n = 0
    for v in reversed(values):
        if condition(v):
            n += 1
        else:
            break
    return n


def strategy_state(name, data):
    cfg = STRATEGIES[name]
    u = data[cfg["underlying"]]
    l = data[cfg["leveraged"]]

    if u.empty or l.empty:
        return {"error": f"{cfg['underlying']} / {cfg['leveraged']} 데이터를 가져오지 못했습니다."}

    snaps = biweekly_snapshots(u, 8)
    if snaps.empty:
        return {"error": "200일 이동평균 데이터가 아직 충분하지 않습니다."}

    latest = snaps.iloc[-1]
    latest_date = snaps.index[-1]
    gap = float(latest["gap"])
    price = float(latest["close"])
    ma = float(latest["ma200"])

    below_count = count_consecutive(snaps["close"].values, lambda x: x < ma if False else False)

    # Use each snapshot's own close vs its own MA.
    below_flags = (snaps["gap"] < 0).tolist()
    above_flags = (snaps["gap"] >= 0).tolist()

    consecutive_below = 0
    for flag in reversed(below_flags):
        if flag:
            consecutive_below += 1
        else:
            break

    consecutive_above = 0
    for flag in reversed(above_flags):
        if flag:
            consecutive_above += 1
        else:
            break

    # Detect whether the current two-week checkpoint is a fresh recross.
    prev_gap = float(snaps.iloc[-2]["gap"]) if len(snaps) >= 2 else np.nan
    recross = bool(gap >= 0 and not np.isnan(prev_gap) and prev_gap < 0)

    # Stage logic:
    # 4 = >= -25% down
    # 3 = 3 biweekly checkpoints below MA OR gap <= stage3 threshold
    # 2 = first below-MA checkpoint
    # 1 = two consecutive biweekly checkpoints above MA
    # Recovery = second consecutive checkpoint after a fresh recross
    if gap <= cfg["stage4_gap"]:
        stage = 4
        stage_name = "4단계 · 시발"
        target = cfg["weights"][4]
        action = f"{cfg['leveraged']} {target}% / 현금 {100-target}%"
    elif gap <= cfg["stage3_gap"] or consecutive_below >= 3:
        stage = 3
        stage_name = "3단계 · 진성 하락"
        target = cfg["weights"][3]
        action = f"{cfg['leveraged']} {target}% / 현금 {100-target}%"
    elif gap < 0:
        stage = 2
        stage_name = "2단계 · 초기 하락"
        target = cfg["weights"][2]
        action = f"{cfg['leveraged']} {target}% / 현금 {100-target}%"
    elif recross or (consecutive_above == 2 and len(snaps) >= 2 and snaps.iloc[-2]["gap"] < 0):
        stage = "recovery"
        stage_name = "상승 복귀 · 2주차"
        target = cfg["weights"]["recovery"]
        action = f"{cfg['leveraged']} {target}% / 현금 {100-target}%"
    elif consecutive_above >= 2:
        stage = 1
        stage_name = "1단계 · 강세장"
        target = cfg["weights"][1]
        action = f"{cfg['leveraged']} {target}% / 현금 {100-target}%"
    else:
        stage = 2
        stage_name = "2단계 · 초기 하락/확인"
        target = cfg["weights"][2]
        action = f"{cfg['leveraged']} {target}% / 현금 {100-target}%"

    # Bubble insurance uses the leveraged ETF's own 200MA.
    lvalid = l.dropna(subset=["ma200"])
    lrow = lvalid.iloc[-1]
    lev_gap = float(lrow["gap"])

    bubble = lev_gap >= cfg["bubble"]
    emergency = None
    for threshold, emergency_weight in cfg["emergency"]:
        if gap <= threshold:
            emergency = {
                "threshold": threshold,
                "weight": emergency_weight,
                "cash": 100 - emergency_weight,
            }
            break

    next_gap = cfg["stage3_gap"] if stage == 2 else cfg["stage4_gap"]
    distance = next_gap - gap

    return {
        "cfg": cfg,
        "snapshots": snaps,
        "date": latest_date,
        "price": price,
        "ma": ma,
        "gap": gap,
        "stage": stage,
        "stage_name": stage_name,
        "target": target,
        "action": action,
        "consecutive_below": consecutive_below,
        "consecutive_above": consecutive_above,
        "recross": recross,
        "leveraged_price": float(lrow["close"]),
        "leveraged_ma": float(lrow["ma200"]),
        "leveraged_gap": lev_gap,
        "bubble": bubble,
        "emergency": emergency,
        "distance": distance,
    }


def fmt_pct(x):
    return f"{x:+.1f}%"


# -----------------------------
# UI
# -----------------------------
st.title("📈 Leveraged Strategy Dashboard V1")
st.caption("매일 데이터 계산 · 2주마다 전략 판정 · 급락 시 월중 비상 대응")

with st.sidebar:
    st.header("설정")
    st.write("판정 주기: **2주**")
    st.write("강세장: 200MA 위 **2회 연속 확인**")
    st.write("진성 하락: 200MA 아래 **3회 연속 확인** 또는 이격도 기준 도달")
    st.write("상승 복귀: 200MA 재돌파 후 **2주차**")
    if st.button("🔄 데이터 새로고침"):
        st.cache_data.clear()
        st.rerun()

data = load_data(["QQQ", "TQQQ", "SOXX", "SOXL"])

states = {}
for name in STRATEGIES:
    states[name] = strategy_state(name, data)

errors = [v["error"] for v in states.values() if "error" in v]
if errors:
    for e in errors:
        st.error(e)
    st.stop()

latest_date = max(v["date"] for v in states.values())
st.info(f"현재 데이터 기준일: **{latest_date:%Y-%m-%d}** · 다음 2주 확인 때 전략 상태를 다시 판정하세요.")

# Top summary
cols = st.columns(2)
for col, (name, s) in zip(cols, states.items()):
    with col:
        st.subheader(name)

        stage_icon = {
            1: "🟢",
            2: "🟡",
            3: "🟠",
            4: "🔴",
            "recovery": "🔵",
        }[s["stage"]]

        st.markdown(f"### {stage_icon} {s['stage_name']}")
        a, b, c = st.columns(3)
        a.metric(s["cfg"]["underlying"], f"${s['price']:,.2f}")
        b.metric("200MA", f"${s['ma']:,.2f}")
        c.metric("200MA 이격", fmt_pct(s["gap"]))

        st.success(f"**현재 목표 비중: {s['cfg']['leveraged']} {s['target']}% / 현금 {100-s['target']}%**")
        st.write(f"**현재 행동:** {s['action']}")

        if s["stage"] == 2:
            st.write(f"200MA 아래 확인: **{s['consecutive_below']}회 연속**")
            st.write(f"3단계 기준까지: **{s['distance']:+.1f}%p**")
        elif s["stage"] == 3:
            st.write(f"200MA 아래 확인: **{s['consecutive_below']}회 연속**")
        elif s["stage"] == 1:
            st.write(f"200MA 위 확인: **{s['consecutive_above']}회 연속**")
        elif s["stage"] == "recovery":
            st.write("재돌파 후 2주차 확인 → 가짜 반등 방지 후 비중 회복")
        else:
            st.write("4단계 방어 구간")

        # Emergency
        if s["emergency"]:
            e = s["emergency"]
            st.error(
                f"🚨 **월중 비상 버튼 발동** · {s['cfg']['underlying']} "
                f"{fmt_pct(s['gap'])} ≤ {e['threshold']:.0f}%\n\n"
                f"즉시 목표: {s['cfg']['leveraged']} **{e['weight']}%** / 현금 **{e['cash']}%**"
            )

        # Bubble insurance
        if s["bubble"]:
            st.warning(
                f"🔥 **버블 보험 발동** · {s['cfg']['leveraged']}가 "
                f"자기 200MA 대비 {fmt_pct(s['leveraged_gap'])}\n\n"
                f"주식 비중을 **60%**까지 낮추는 것을 검토"
            )
        else:
            st.caption(
                f"버블 보험: {s['cfg']['leveraged']} 200MA 이격 "
                f"{fmt_pct(s['leveraged_gap'])} / 발동 기준 +{s['cfg']['bubble']:.0f}%"
            )

st.divider()

# Decision table
st.subheader("🎯 한눈에 보는 실행 규칙")
rows = []
for name, s in states.items():
    rows.append({
        "전략": name,
        "현재 단계": s["stage_name"],
        "기초지수": s["cfg"]["underlying"],
        "기초지수 / 200MA": fmt_pct(s["gap"]),
        "레버리지 ETF": s["cfg"]["leveraged"],
        "목표 ETF 비중": f"{s['target']}%",
        "목표 현금": f"{100-s['target']}%",
        "2주 판정": f"위 {s['consecutive_above']}회 / 아래 {s['consecutive_below']}회",
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# History
st.subheader("📊 최근 2주 판정 이력")

tab1, tab2 = st.tabs(["TQQQ / QQQ", "SOXL / SOXX"])
for tab, name in zip([tab1, tab2], STRATEGIES):
    with tab:
        s = states[name]
        h = s["snapshots"].copy()
        h["날짜"] = h.index.strftime("%Y-%m-%d")
        h["종가"] = h["close"].map(lambda x: f"${x:,.2f}")
        h["200MA"] = h["ma200"].map(lambda x: f"${x:,.2f}")
        h["이격"] = h["gap"].map(lambda x: f"{x:+.1f}%")
        h["위/아래"] = np.where(h["gap"] >= 0, "위", "아래")
        h["아래 연속"] = ""
        h["위 연속"] = ""

        below = 0
        above = 0
        below_vals = []
        above_vals = []
        for flag in (h["gap"] < 0):
            below = below + 1 if flag else 0
            below_vals.append(below)
        for flag in (h["gap"] >= 0):
            above = above + 1 if flag else 0
            above_vals.append(above)
        h["아래 연속"] = below_vals
        h["위 연속"] = above_vals

        st.dataframe(
            h[["날짜", "종가", "200MA", "이격", "위/아래", "아래 연속", "위 연속"]].iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

st.divider()

st.subheader("🧭 V1 판정 원칙")
st.markdown("""
- **데이터:** 일봉 종가와 200일 이동평균을 매일 계산
- **정규 판정:** 사용자가 **2주마다** 확인
- **1단계:** 200MA 위 2회 연속 확인
- **2단계:** 200MA 첫 이탈
- **3단계:** 200MA 아래 3회 연속 확인 **또는** 이격도 기준 도달
- **4단계:** 200MA 대비 -25% 이하
- **상승 복귀:** 200MA 재돌파 후 2주차 확인
- **비상 버튼:** 급락 기준에 도달하면 2주 판정일을 기다리지 않고 즉시 대응
- **버블 보험:** 레버리지 ETF 자체가 자기 200MA 대비 과열 기준을 넘으면 별도 경고
""")

st.caption("※ 본 대시보드는 사용자가 정한 규칙을 기계적으로 표시하는 도구이며 투자 조언이 아닙니다.")
