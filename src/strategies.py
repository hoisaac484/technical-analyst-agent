# src/strategies.py
from __future__ import annotations

from typing import Dict, Callable, List
import numpy as np
import pandas as pd


# -----------------------------
# Regime snapshot (long-only)
# -----------------------------
def regime_snapshot(df_feat: pd.DataFrame, cfg: dict) -> dict:
    """
    Classify the current market regime using the latest available data.

    Returns:
      {
        "regime": "RANGE" | "UPTREND" | "UNCLEAR",
        "flags": [...],
        "evidence": {...}
      }
    """
    if df_feat is None or df_feat.empty:
        return {"regime": "UNCLEAR", "flags": ["NO_DATA"], "evidence": {}}

    last = df_feat.iloc[-1]

    adx = float(last.get("adx14", np.nan))
    sma50 = float(last.get("sma50", np.nan))
    sma200 = float(last.get("sma200", np.nan))
    vol = float(last.get("vol_20d", np.nan))

    flags: List[str] = []

    # Feature availability
    need = ["adx14", "sma50", "sma200"]
    if any(pd.isna(last.get(c, np.nan)) for c in need):
        flags.append("INSUFFICIENT_FEATURES")

    adx_range_max = float(cfg.get("adx_range_max", 20))
    adx_trend_min = float(cfg.get("adx_trend_min", 25))

    if "INSUFFICIENT_FEATURES" in flags:
        regime = "UNCLEAR"
    else:
        if adx < adx_range_max:
            regime = "RANGE"
        elif adx >= adx_trend_min and sma50 > sma200:
            regime = "UPTREND"
        else:
            regime = "UNCLEAR"

    return {
        "regime": regime,
        "flags": flags,
        "evidence": {
            "adx14": adx,
            "sma50": sma50,
            "sma200": sma200,
            "sma50_gt_sma200": bool(sma50 > sma200)
            if not (np.isnan(sma50) or np.isnan(sma200))
            else None,
            "vol_20d": vol,
        },
    }


# -----------------------------
# Strategy selection (sticky)
# -----------------------------
def propose_strategy(regime_info: dict, cfg: dict) -> str:
    """
    Select a strategy when:
      - no active strategy exists, OR
      - the current strategy has been invalidated.

    Mapping:
      RANGE   -> mean_reversion
      UPTREND -> trend_follow
      UNCLEAR -> cash
    """
    regime = regime_info.get("regime", "UNCLEAR")
    flags = set(regime_info.get("flags", []))

    if "INSUFFICIENT_FEATURES" in flags:
        return "cash"

    if regime == "RANGE":
        return "mean_reversion"

    if regime == "UPTREND":
        return "trend_follow"

    return "cash"


def is_strategy_valid(active_strategy: str, df_feat: pd.DataFrame, cfg: dict) -> bool:
    """
    Decide whether the currently active strategy remains valid.
    Strategy switching is event-driven, not daily.
    """
    reg = regime_snapshot(df_feat, cfg)
    regime = reg["regime"]
    flags = set(reg["flags"])
    last = df_feat.iloc[-1]

    adx = float(last.get("adx14", np.nan))
    sma50 = float(last.get("sma50", np.nan))
    sma200 = float(last.get("sma200", np.nan))

    adx_trend_min = float(cfg.get("adx_trend_min", 25))

    if "INSUFFICIENT_FEATURES" in flags:
        return active_strategy == "cash"

    if active_strategy == "mean_reversion":
        # Mean reversion invalidated when a trend emerges
        if not np.isnan(adx) and adx >= adx_trend_min:
            return False
        return True

    if active_strategy == "trend_follow":
        # Trend follow invalidated when the trend breaks
        if np.isnan(sma50) or np.isnan(sma200):
            return False
        return sma50 > sma200

    if active_strategy == "cash":
        # Cash is invalid once a clear actionable regime appears
        return regime == "UNCLEAR"

    return False


# -----------------------------
# Human-readable rules
# -----------------------------
def explain_rules(strategy_name: str, cfg: dict) -> str:
    if strategy_name == "mean_reversion":
        rsi_entry = float(cfg.get("rsi_entry", 35))
        mr_atr_k = float(cfg.get("mr_atr_stop_k", 2.0))
        mr_max_hold = int(cfg.get("mr_max_hold", 10))
        return (
            "Mean Reversion (Long-only)\n"
            f"- Enter LONG when Close < Lower Bollinger Band AND RSI(14) < {rsi_entry:.0f}.\n"
            "- Exit when price reverts to the Bollinger mid-band, OR\n"
            f"  when price falls below Entry − {mr_atr_k:.1f}×ATR(14), OR\n"
            f"  after {mr_max_hold} trading days."
        )

    if strategy_name == "trend_follow":
        return (
            "Trend Follow (Long-only)\n"
            "- Hold LONG when SMA(50) > SMA(200).\n"
            "- Exit to CASH when SMA(50) ≤ SMA(200)."
        )

    return "Cash\n- Stay in CASH (no position)."


# -----------------------------
# Signal generators (0/1)
# -----------------------------
def signal_cash(df_feat: pd.DataFrame, cfg: dict) -> pd.Series:
    return pd.Series(0, index=df_feat.index, dtype=int)


def signal_trend_follow(df_feat: pd.DataFrame, cfg: dict) -> pd.Series:
    cond = df_feat["sma50"] > df_feat["sma200"]
    sig = cond.astype(int)
    sig = sig.where(~sig.isna(), 0).astype(int)
    return sig


def signal_mean_reversion(df_feat: pd.DataFrame, cfg: dict) -> pd.Series:
    rsi_entry = float(cfg.get("rsi_entry", 35))
    mr_atr_k = float(cfg.get("mr_atr_stop_k", 2.0))
    mr_max_hold = int(cfg.get("mr_max_hold", 10))

    sig = pd.Series(0, index=df_feat.index, dtype=int)

    in_pos = False
    entry_price = np.nan
    entry_i = -1

    for i in range(len(df_feat)):
        row = df_feat.iloc[i]

        close = row.get("close", np.nan)
        rsi = row.get("rsi14", np.nan)
        bb_lower = row.get("bb_lower", np.nan)
        bb_mid = row.get("bb_mid", np.nan)
        atr = row.get("atr14", np.nan)

        if any(pd.isna(x) for x in [close, rsi, bb_lower, bb_mid, atr]):
            sig.iloc[i] = 1 if in_pos else 0
            continue

        if not in_pos:
            if close < bb_lower and rsi < rsi_entry:
                in_pos = True
                entry_price = float(close)
                entry_i = i
                sig.iloc[i] = 1
            else:
                sig.iloc[i] = 0
        else:
            held = i - entry_i
            stop_price = entry_price - mr_atr_k * atr

            exit_cond = (
                close >= bb_mid
                or close <= stop_price
                or held >= mr_max_hold
            )

            if exit_cond:
                in_pos = False
                entry_price = np.nan
                entry_i = -1
                sig.iloc[i] = 0
            else:
                sig.iloc[i] = 1

    return sig


# -----------------------------
# Strategy registry
# -----------------------------
STRATEGY_MAP: Dict[str, Callable[[pd.DataFrame, dict], pd.Series]] = {
    "mean_reversion": signal_mean_reversion,
    "trend_follow": signal_trend_follow,
    "cash": signal_cash,
}
