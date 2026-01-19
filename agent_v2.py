"""
KO Mean-Reversion Strategy — Backtest (Virtual Capital + Fixed-Fractional Risk)

Key features:
✅ Unlimited cash/margin blocking (virtual equity compounding)
✅ Position sizing by fixed-fractional risk (risk_pct of equity per entry)
✅ Supports multiple simultaneous trades:
      - max_open_trades = 1        -> single position
      - max_open_trades = N (int)  -> cap at N
      - max_open_trades = False/None -> unlimited
✅ Tracks:
      - equity (mark-to-market)
      - gross_leverage = sum(|notional|) / mark-to-market equity
      - open_trades count
✅ One-window plotting with separate subplots (no twin axis)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import os
from dotenv import load_dotenv

from openai import AzureOpenAI

load_dotenv()  # ← THIS is mandatory



# -----------------------------
# Data helpers
# -----------------------------
def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure single-level OHLCV columns from yfinance output (handles MultiIndex)."""
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance often returns (field, ticker)
        tickers = list(dict.fromkeys([c[1] for c in df.columns]))
        t0 = tickers[0]
        df = df.xs(t0, axis=1, level=1, drop_level=True)

    df.columns = [str(c).title() for c in df.columns]
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Got: {list(df.columns)}")

    df = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
    return df


# -----------------------------
# Indicators
# -----------------------------
def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return rma(tr, length)


def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    rs = rma(up, length) / rma(down, length)
    return 100 - (100 / (1 + rs))


def smma(series: pd.Series, length: int) -> pd.Series:
    # Wilder-style smoothing ~= RMA
    return rma(series, length)


def cmo(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(length).sum()
    down = (-delta).clip(lower=0).rolling(length).sum()
    return 100 * (up - down) / (up + down)


def supertrend(df: pd.DataFrame, period: int = 1, multiplier: float = 0.1) -> pd.DataFrame:
    """
    Minimal Supertrend:
      - direction: 1 bullish, -1 bearish
      - buy_signal: bearish -> bullish flip
    """
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    _atr = atr(df, period)
    hl2 = (h + l) / 2.0
    upperband = hl2 + multiplier * _atr
    lowerband = hl2 - multiplier * _atr

    final_upper = upperband.copy()
    final_lower = lowerband.copy()

    for i in range(1, len(df)):
        if pd.isna(_atr.iat[i]):
            continue

        if (upperband.iat[i] < final_upper.iat[i - 1]) or (c.iat[i - 1] > final_upper.iat[i - 1]):
            final_upper.iat[i] = upperband.iat[i]
        else:
            final_upper.iat[i] = final_upper.iat[i - 1]

        if (lowerband.iat[i] > final_lower.iat[i - 1]) or (c.iat[i - 1] < final_lower.iat[i - 1]):
            final_lower.iat[i] = lowerband.iat[i]
        else:
            final_lower.iat[i] = final_lower.iat[i - 1]

    st = pd.Series(index=df.index, dtype="float64")
    direction = pd.Series(index=df.index, dtype="int64")

    direction.iat[0] = 1
    st.iat[0] = final_lower.iat[0]

    for i in range(1, len(df)):
        prev_dir = int(direction.iat[i - 1])
        prev_st = float(st.iat[i - 1])

        if prev_dir == 1:
            if c.iat[i] < final_lower.iat[i]:
                direction.iat[i] = -1
                st.iat[i] = float(final_upper.iat[i])
            else:
                direction.iat[i] = 1
                st.iat[i] = float(max(final_lower.iat[i], prev_st))
        else:
            if c.iat[i] > final_upper.iat[i]:
                direction.iat[i] = 1
                st.iat[i] = float(final_lower.iat[i])
            else:
                direction.iat[i] = -1
                st.iat[i] = float(min(final_upper.iat[i], prev_st))

    buy_signal = (direction.shift(1) == -1) & (direction == 1)
    return pd.DataFrame({"st": st, "direction": direction, "buy_signal": buy_signal})


# -----------------------------
# Backtest (virtual capital + fixed-fractional risk, multi-trade capable)
# -----------------------------
@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    shares: float
    sl: float
    tp: float
    risk_amount: float  # equity * risk_pct at entry (for R-multiple)
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    r_mult: Optional[float] = None
    outcome: Optional[str] = None  # TP / SL / EOD


def compute_drawdown(curve: pd.Series) -> pd.Series:
    peak = curve.cummax()
    return curve / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    dd = compute_drawdown(equity)
    return float(dd.min()) if len(dd) else float("nan")


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float((r.mean() / r.std(ddof=1)) * math.sqrt(periods_per_year))


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return float("nan")
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


MaxOpenTradesType = Union[int, bool, None]


def run_backtest_virtual_capital(
    df_in: pd.DataFrame,
    *,
    initial_equity: float = 1.0,
    risk_pct: float = 0.01,
    max_open_trades: MaxOpenTradesType = 1,  # int cap, or False/None for unlimited
    allow_fractional_shares: bool = True,
    atr_len: int = 100,
    sl_atr_mult: float = 4.0,
    rr: float = 1.4,
    commission_rate: float = 0.001,  # 0.001 per trade
    slippage_bps: float = 0.0,
    dist_threshold_pct: float = 0.5,
    candle_height_limit_pct: float = 2.0,
) -> Dict[str, Any]:
    df = normalize_ohlc(df_in)

    # Typical prices (matching your strategy)
    hlc3 = (df["High"] + df["Low"] + df["Close"]) / 3.0
    hlcc4 = (df["High"] + df["Low"] + df["Close"] + df["Close"]) / 4.0

    # Indicators
    st = supertrend(df, period=1, multiplier=0.1)
    df["buy_signal"] = st["buy_signal"]
    df["rsi50"] = rsi(hlc3, 50)

    df["ma200"] = smma(df["Close"], 200)
    df["dist_pct"] = (df["Close"] / df["ma200"] - 1.0) * 100.0
    df["dist_ok"] = df["dist_pct"].abs() > dist_threshold_pct

    df["cmo15"] = cmo(hlcc4, 15)
    df["cmo_smma4"] = smma(df["cmo15"], 4)
    df["cmo_ok"] = df["cmo15"] > df["cmo_smma4"]

    df["candle_ht_pct"] = (df["High"] - df["Low"]) / df["Close"] * 100.0
    df["candle_ok"] = df["candle_ht_pct"] <= candle_height_limit_pct

    df["atr"] = atr(df, atr_len)

    # Entry condition (same as your logic)
    df["entry_cond"] = (
        df["buy_signal"].fillna(False)
        & (df["rsi50"] < 50)
        & df["dist_ok"].fillna(False)
        & df["cmo_ok"].fillna(False)
        & df["candle_ok"].fillna(False)
    )

    def apply_slippage(px: float, side: str) -> float:
        adj = px * (slippage_bps / 10000.0)
        return px + adj if side == "buy" else px - adj

    UNLIMITED = max_open_trades in (False, None)
    max_open_trades_int = int(max_open_trades) if not UNLIMITED else None
    if (not UNLIMITED) and max_open_trades_int is not None and max_open_trades_int < 1:
        raise ValueError("max_open_trades must be >= 1, or set to False/None for unlimited.")

    equity = float(initial_equity)
    trades: List[Trade] = []
    open_trades: List[Trade] = []

    equity_curve: List[Tuple[pd.Timestamp, float]] = []
    gross_leverage_curve: List[Tuple[pd.Timestamp, float]] = []
    open_trades_curve: List[Tuple[pd.Timestamp, int]] = []

    idx = df.index.to_list()

    for i in range(len(df) - 1):  # entries at next bar open
        t = idx[i]
        row = df.iloc[i]

        bar_low = float(row["Low"])
        bar_high = float(row["High"])

        # 1) Exit logic (SL/TP in current bar)
        closed: List[Trade] = []
        for tr in open_trades:
            hit_sl = bar_low <= tr.sl
            hit_tp = bar_high >= tr.tp
            if not (hit_sl or hit_tp):
                continue

            # Conservative if both hit: SL first
            if hit_sl and hit_tp:
                exit_px = tr.sl
                outcome = "SL"
            elif hit_sl:
                exit_px = tr.sl
                outcome = "SL"
            else:
                exit_px = tr.tp
                outcome = "TP"

            exit_px = apply_slippage(float(exit_px), "sell")
            exit_notional = exit_px * tr.shares
            exit_fee = exit_notional * commission_rate
            pnl = (exit_px - tr.entry_price) * tr.shares - exit_fee


            equity += float(pnl)

            tr.exit_time = t
            tr.exit_price = float(exit_px)
            tr.pnl = float(pnl)
            tr.outcome = outcome
            tr.r_mult = (float(pnl) / tr.risk_amount) if tr.risk_amount > 0 else float("nan")
            closed.append(tr)

        if closed:
            open_trades = [tr for tr in open_trades if tr not in closed]
            trades.extend(closed)

        # 2) Entry logic (next open) with cap/unlimited
        slots_ok = UNLIMITED or (len(open_trades) < int(max_open_trades_int))  # type: ignore[arg-type]
        can_enter = bool(row["entry_cond"]) and (not pd.isna(row["atr"])) and (equity > 0) and slots_ok

        if can_enter:
            next_open = float(df.iloc[i + 1]["Open"])
            entry_px = apply_slippage(next_open, "buy")
            a = float(row["atr"])

            sl = entry_px - sl_atr_mult * a
            tp = entry_px + (sl_atr_mult * rr) * a

            risk_per_share = entry_px - sl
            if risk_per_share > 0:
                risk_amount = equity * risk_pct
                shares = risk_amount / risk_per_share

                if not allow_fractional_shares:
                    shares = math.floor(shares)

                if shares > 0:
                    entry_notional = entry_px * shares
                    entry_fee = entry_notional * commission_rate
                    equity -= float(entry_fee)


                    open_trades.append(
                        Trade(
                            entry_time=idx[i + 1],
                            entry_price=float(entry_px),
                            shares=float(shares),
                            sl=float(sl),
                            tp=float(tp),
                            risk_amount=float(risk_amount),
                        )
                    )

        # 3) Mark-to-market + leverage + open-trades tracking
        close_px = float(row["Close"])

        unreal = 0.0
        notional = 0.0
        for tr in open_trades:
            unreal += (close_px - tr.entry_price) * tr.shares
            notional += abs(close_px * tr.shares)

        mtm_equity = float(equity + unreal)
        equity_curve.append((t, mtm_equity))

        gl = float(notional / mtm_equity) if mtm_equity > 0 else float("nan")
        gross_leverage_curve.append((t, gl))

        open_trades_curve.append((t, int(len(open_trades))))

    # 4) Close remaining at last close
    last_t = df.index[-1]
    last_close = apply_slippage(float(df.iloc[-1]["Close"]), "sell")
    for tr in open_trades:
        exit_notional = last_close * tr.shares
        exit_fee = exit_notional * commission_rate
        pnl = (last_close - tr.entry_price) * tr.shares - exit_fee

        equity += float(pnl)

        tr.exit_time = last_t
        tr.exit_price = float(last_close)
        tr.pnl = float(pnl)
        tr.outcome = "EOD"
        tr.r_mult = (float(pnl) / tr.risk_amount) if tr.risk_amount > 0 else float("nan")
        trades.append(tr)

    equity_series = pd.Series([v for _, v in equity_curve], index=[d for d, _ in equity_curve], name="equity")
    gross_lev_series = pd.Series(
        [v for _, v in gross_leverage_curve],
        index=[d for d, _ in gross_leverage_curve],
        name="gross_leverage",
    )
    open_trades_series = pd.Series(
        [v for _, v in open_trades_curve],
        index=[d for d, _ in open_trades_curve],
        name="open_trades",
    )

    rets = equity_series.pct_change()

    wins = [tr for tr in trades if tr.pnl is not None and tr.pnl > 0]
    losses = [tr for tr in trades if tr.pnl is not None and tr.pnl < 0]
    gross_profit = float(sum(tr.pnl for tr in wins)) if wins else 0.0
    gross_loss = float(-sum(tr.pnl for tr in losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    r_list = [tr.r_mult for tr in trades if tr.r_mult is not None and math.isfinite(tr.r_mult)]
    avg_r = float(np.mean(r_list)) if r_list else float("nan")

    observed_max_open_trades = int(open_trades_series.max()) if len(open_trades_series) else 0
    observed_max_leverage = float(np.nanmax(gross_lev_series.values)) if len(gross_lev_series) else float("nan")

    max_open_trades_repr: Union[int, str] = "unlimited" if UNLIMITED else int(max_open_trades_int)  # type: ignore[arg-type]

    metrics = {
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
        "bars": int(len(df)),
        "initial_equity": float(initial_equity),
        "final_equity": float(equity_series.iloc[-1]) if len(equity_series) else float("nan"),
        "num_trades": int(len(trades)),
        "win_rate": (len(wins) / len(trades)) if trades else float("nan"),
        "profit_factor": float(profit_factor) if math.isfinite(profit_factor) else float("inf"),
        "avg_R": avg_r,
        "CAGR": cagr(equity_series),
        "Sharpe": sharpe_ratio(rets, 252),
        "MaxDrawdown": max_drawdown(equity_series) if len(equity_series) else float("nan"),
        "risk_pct": float(risk_pct),
        "max_open_trades": max_open_trades_repr,
        "fractional_shares": bool(allow_fractional_shares),
        "observed_max_open_trades": observed_max_open_trades,
        "observed_max_leverage": observed_max_leverage,
    }

    return {
        "metrics": metrics,
        "equity": equity_series,
        "gross_leverage": gross_lev_series,
        "open_trades": open_trades_series,
        "trades": trades,
        "df": df,
    }


# -----------------------------
# Plotting helpers
# -----------------------------
def print_metrics(m: Dict[str, Any]) -> None:
    def pct(x: float) -> str:
        return "nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.2%}"

    def num(x: float) -> str:
        return "nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.4f}"

    print("\n=== Metrics (Virtual Capital) ===")
    print(f"{'start':>28}: {m['start']}")
    print(f"{'end':>28}: {m['end']}")
    print(f"{'bars':>28}: {m['bars']}")
    print(f"{'risk_pct':>28}: {pct(m['risk_pct'])}")
    print(f"{'max_open_trades(set)':>28}: {m['max_open_trades']}")
    print(f"{'max_open_trades(obs)':>28}: {m.get('observed_max_open_trades', 'na')}")
    print(f"{'max_leverage(obs)':>28}: {num(m.get('observed_max_leverage', float('nan')))}")
    print(f"{'fractional_shares':>28}: {m['fractional_shares']}")
    print(f"{'initial_equity':>28}: {m['initial_equity']:.4f}")
    print(f"{'final_equity':>28}: {m['final_equity']:.4f}")
    print(f"{'num_trades':>28}: {m['num_trades']}")
    print(f"{'win_rate':>28}: {pct(m['win_rate'])}")
    pf = m["profit_factor"]
    print(f"{'profit_factor':>28}: {'inf' if pf == float('inf') else f'{pf:.3f}'}")
    print(f"{'avg_R':>28}: {num(m['avg_R'])}")
    print(f"{'CAGR':>28}: {pct(m['CAGR'])}")
    print(f"{'Sharpe':>28}: {num(m['Sharpe'])}")
    print(f"{'MaxDrawdown':>28}: {pct(m['MaxDrawdown'])}")



def build_trade_note_prompt(result: Dict[str, Any]) -> str:
    """
    Build a prompt grounded on your backtest metrics + the latest signal state.
    Keep it structured so the output is consistent and assessable.
    """
    m = result["metrics"]
    df = result["df"].copy()

    # Grab latest row to describe current regime/signal status
    last = df.iloc[-1]
    latest = {
        "date": str(df.index[-1].date()),
        "close": float(last["Close"]),
        "buy_signal": bool(last.get("buy_signal", False)),
        "rsi50": float(last.get("rsi50", float("nan"))),
        "dist_pct": float(last.get("dist_pct", float("nan"))),
        "cmo15": float(last.get("cmo15", float("nan"))),
        "cmo_smma4": float(last.get("cmo_smma4", float("nan"))),
        "candle_ht_pct": float(last.get("candle_ht_pct", float("nan"))),
        "atr": float(last.get("atr", float("nan"))),
        "entry_cond": bool(last.get("entry_cond", False)),
    }

    # Metrics we want explicitly in the note
    # (adjust keys to match your metrics dict)
    metrics_block = {
        "period": f'{m.get("start")} → {m.get("end")}',
        "final_equity": m.get("final_equity"),
        "CAGR": m.get("CAGR"),
        "Sharpe": m.get("Sharpe"),
        "MaxDrawdown": m.get("MaxDrawdown"),
        "num_trades": m.get("num_trades"),
        "win_rate": m.get("win_rate"),
        "profit_factor": m.get("profit_factor"),
        "avg_R": m.get("avg_R"),
        "observed_max_open_trades": m.get("observed_max_open_trades"),
        "observed_max_leverage": m.get("observed_max_leverage"),
        "risk_pct": m.get("risk_pct"),
        "max_open_trades_setting": m.get("max_open_trades"),
    }

    return f"""
You are a buy-side Technical Analyst Agent in an asset management team.
Write a professional 1–2 page TRADE NOTE.

Hard requirements:
- Use the backtest metrics provided (CAGR, Sharpe, max drawdown, win rate/hit rate, number of trades).
- Provide a clear recommendation: BUY / HOLD / SELL (or "Long / Flat" if you prefer).
- Provide a concrete trade plan: entry trigger, stop loss, take profit, sizing logic, and key risks.
- Explain signal logic briefly (why the strategy enters/exits).
- Be transparent about limitations (data, regime dependence, costs, survivorship, etc.).
- Keep it structured with headings and bullet points where useful.

Backtest metrics (use these numbers, don't invent):
{metrics_block}

Latest market snapshot (use these numbers, don't invent):
{latest}

Output format (use headings):
1) Executive Summary (recommendation + 2–3 key reasons)
2) Strategy Logic (indicators + entry/exit conditions)
3) Backtest Results (table-like bullets + interpretation)
4) Current Signal & Trade Plan (entry/SL/TP/sizing; include ATR-based distances if relevant)
5) Risk Management & Limitations
""".strip()


def generate_trade_note_azure(result: Dict[str, Any]) -> str:
    """
    Calls Azure OpenAI to generate the trade note text.
    """
    endpoint = os.environ["AZURE_ENDPOINT"]
    api_key = os.environ["AZURE_API_KEY"]
    api_version = os.environ.get("AZURE_API_VERSION", "2024-02-15-preview")
    deployment = os.environ["AZURE_DEPLOYMENT_NAME"]  # Azure deployment name

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    prompt = build_trade_note_prompt(result)

    resp = client.chat.completions.create(
        model=deployment,  # Azure: deployment name
        messages=[
            {"role": "system", "content": "You write concise, data-grounded institutional research notes."},
            {"role": "user", "content": prompt},
        ],
        temperature=1,
    )
    return resp.choices[0].message.content

def plot_all_in_one_window(result: Dict[str, Any]) -> None:
    df = result["df"]

    equity = result["equity"].dropna()
    idx = equity.index
    if len(idx) < 5:
        raise ValueError("Not enough data to plot.")

    price = df.loc[idx, "Close"].astype(float)

    # Normalize (so initial capital doesn't flatten things)
    price_norm = price / price.iloc[0]
    equity_norm = equity / equity.iloc[0]

    # Buy&Hold equity (normalized)
    bh_equity = (price / price.iloc[0]).rename("buy_hold_equity")

    # Drawdowns
    dd_bh = compute_drawdown(bh_equity)
    dd_strat = compute_drawdown(equity_norm.rename("strategy_equity"))

    # Leverage + open trades (separate plots)
    lev = result["gross_leverage"].reindex(idx).astype(float)
    ot = result["open_trades"].reindex(idx).astype(float)

    max_ot_set = result["metrics"].get("max_open_trades", "unlimited")
    max_lev_obs = float(result["metrics"].get("observed_max_leverage", float("nan")))

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(12, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2, 1]},
    )

    # 1) Price vs Equity (normalized)
    ax = axes[0]
    ax.plot(price_norm, label="KO Buy & Hold (norm)")
    ax.plot(equity_norm, label="Strategy Equity (norm)")
    ax.set_title("Strategy vs Buy & Hold (Normalized)")
    ax.set_ylabel("Normalized value")
    ax.grid(True)
    ax.legend(loc="upper left")

    # 2) Drawdowns
    ax = axes[1]
    ax.plot(dd_bh, label="B&H drawdown")
    ax.plot(dd_strat, label="Strategy drawdown")
    ax.axhline(0, linewidth=1)
    ax.set_title("Drawdown Comparison")
    ax.set_ylabel("Drawdown")
    ax.grid(True)
    ax.legend(loc="lower left")

    # 3) Leverage
    ax = axes[2]
    ax.plot(lev, label="Gross Leverage")
    ax.axhline(1.0, linewidth=1, label="1x")
    if math.isfinite(max_lev_obs):
        ax.axhline(max_lev_obs, linestyle="--", linewidth=1, label=f"Max (obs) = {max_lev_obs:.3f}x")
    ax.set_title("Gross Leverage")
    ax.set_ylabel("Leverage (x)")
    ax.grid(True)
    ax.legend(loc="upper left")

    # 4) Open trades
    ax = axes[3]
    ax.step(ot.index, ot.values, where="post", label="Open Trades")
    if max_ot_set != "unlimited":
        ax.axhline(int(max_ot_set), linestyle="--", linewidth=1, label=f"Max Open Trades (set) = {max_ot_set}")
    ax.set_title("Open Trades")
    ax.set_ylabel("Count")
    ax.set_xlabel("Date")
    ax.grid(True)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()



# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    ticker = "KO"
    START_DATE = "2016-01-15"
    END_DATE = "2026-01-15"

    raw = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("No data returned from yfinance. Check internet/ticker.")

    df = normalize_ohlc(raw)

    # Examples:
    #   max_open_trades=1       -> single position
    #   max_open_trades=3       -> cap at 3
    #   max_open_trades=False   -> unlimited
    result = run_backtest_virtual_capital(
        df,
        max_open_trades=False,
        atr_len=100,
        sl_atr_mult=4.0,
        rr=1.4,
        risk_pct=0.01,
        slippage_bps=0.0,
        dist_threshold_pct=0.5,
        candle_height_limit_pct=2.0,
    )


    print_metrics(result["metrics"])
    plot_all_in_one_window(result)
    note = generate_trade_note_azure(result)
    print("\n\n=== TRADE NOTE (LLM) ===\n")
    print(note)
