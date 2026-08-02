"""Point-in-time feature engineering for the cross-sectional model."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd

from .config import BINARY_FEATURES, CONTINUOUS_FEATURES, BacktestConfig
from .data import adjust_ohlc
from .indicators import adx, compute_volume_profile_levels, ema, rolling_beta, triple_barrier_meta_label, true_range


def compute_features_for_symbol(
    df_raw: pd.DataFrame,
    benchmark: pd.DataFrame,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    df = adjust_ohlc(df_raw)
    df = df.set_index("date").sort_index()
    bench = benchmark.set_index("date").sort_index()
    bench_close = bench["adj_close"].reindex(df.index).ffill()

    close = df["adj_close"].astype(float)
    open_ = df["adj_open"].astype(float)
    high = df["adj_high"].astype(float)
    low = df["adj_low"].astype(float)
    volume = df["volume"].astype(float).fillna(0.0)
    typical = (high + low + close) / 3.0
    ret = close.pct_change()
    bench_ret = bench_close.pct_change()

    out = pd.DataFrame(index=df.index)
    out["symbol"] = df["symbol"].iloc[0]
    out["open"] = open_
    out["close"] = close
    out["volume"] = volume
    out["turnover_tl"] = close * volume

    tr = true_range(high, low, close)
    out["atr14"] = tr.rolling(14).mean()
    out["ema20"] = ema(close, 20)
    out["ema50"] = ema(close, 50)
    out["ema200"] = ema(close, 200)
    out["close_above_ema20"] = (close > out["ema20"]).astype(float)
    out["ema_stack"] = ((close > out["ema20"]) & (out["ema20"] > out["ema50"])).astype(float)

    out["adx14"] = adx(high, low, close, 14)
    out["roc21"] = close / close.shift(21) - 1.0
    out["mom63_5"] = close.shift(5) / close.shift(63) - 1.0
    out["mom126_21"] = close.shift(21) / close.shift(126) - 1.0
    out["rolling_sharpe21"] = ret.rolling(21).mean() / ret.rolling(21).std().replace(0, np.nan)

    beta63 = rolling_beta(ret, bench_ret, 63)
    stock_ret21 = close / close.shift(21) - 1.0
    bench_ret21 = bench_close / bench_close.shift(21) - 1.0
    out["ralpha21"] = stock_ret21 - beta63 * bench_ret21

    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    out["obv_slope20"] = (obv - obv.shift(20)) / volume.rolling(20).sum().replace(0, np.nan)

    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mfm.fillna(0.0) * volume
    out["cmf21"] = mfv.rolling(21).sum() / volume.rolling(21).sum().replace(0, np.nan)
    out["rel_volume20"] = volume / volume.rolling(20).mean().replace(0, np.nan)
    out["turnover_rank_feature"] = np.log1p(out["turnover_tl"])

    pv = typical * volume
    out["vwap20"] = pv.rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    out["vwap63"] = pv.rolling(63).sum() / volume.rolling(63).sum().replace(0, np.nan)
    out["vwap20_slope5"] = out["vwap20"] / out["vwap20"].shift(5) - 1.0
    out["vwap63_slope5"] = out["vwap63"] / out["vwap63"].shift(5) - 1.0
    out["dist_vwap20_atr"] = (close - out["vwap20"]) / out["atr14"].replace(0, np.nan)
    out["dist_vwap63_atr"] = (close - out["vwap63"]) / out["atr14"].replace(0, np.nan)
    out["close_above_vwap20"] = (close > out["vwap20"]).astype(float)
    out["close_above_vwap63"] = (close > out["vwap63"]).astype(float)

    poc, vah, val = compute_volume_profile_levels(
        typical.to_numpy(dtype=float),
        high.to_numpy(dtype=float),
        low.to_numpy(dtype=float),
        volume.to_numpy(dtype=float),
        cfg.profile_window,
        cfg.profile_bins,
    )
    out["prior_poc"] = poc
    out["prior_vah"] = vah
    out["prior_val"] = val
    out["dist_poc_atr"] = (close - out["prior_poc"]) / out["atr14"].replace(0, np.nan)
    out["value_position"] = (close - out["prior_val"]) / (out["prior_vah"] - out["prior_val"]).replace(0, np.nan)
    out["profile_width_atr"] = (out["prior_vah"] - out["prior_val"]) / out["atr14"].replace(0, np.nan)
    out["poc_migration5_atr"] = (out["prior_poc"] - out["prior_poc"].shift(5)) / out["atr14"].replace(0, np.nan)
    out["close_above_poc"] = (close > out["prior_poc"]).astype(float)
    out["close_above_vah"] = (close > out["prior_vah"]).astype(float)
    out["accepted_above_vah"] = ((close > out["prior_vah"]) & (close.shift(1) > out["prior_vah"].shift(1))).astype(float)
    out["failed_vah_breakout"] = ((high > out["prior_vah"]) & (close < out["prior_vah"]) & (close < out["vwap20"])).astype(float)

    out["next_ret_1d"] = close.pct_change().shift(-1)
    fwd = close.shift(-cfg.holding_label_days) / close - 1.0
    bench_fwd = bench_close.shift(-cfg.holding_label_days) / bench_close - 1.0
    future_vol = ret.shift(-1).iloc[::-1].rolling(cfg.holding_label_days).std().iloc[::-1]
    future_vol = future_vol * math.sqrt(cfg.holding_label_days)
    out["label_fwd_excess"] = fwd - bench_fwd
    out["label_ml"] = out["label_fwd_excess"] / future_vol.replace(0, np.nan)
    out["label_meta"] = triple_barrier_meta_label(
        close=close,
        high=high,
        low=low,
        atr=out["atr14"],
        horizon=cfg.holding_label_days,
        profit_atr=cfg.meta_profit_atr,
        stop_atr=cfg.meta_stop_atr,
    )
    out["raw_fwd_return"] = fwd
    return out.reset_index().rename(columns={"index": "date"})


def make_feature_panel(features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    panel = pd.concat(features.values(), ignore_index=True)
    panel = panel.replace([np.inf, -np.inf], np.nan)

    ranked_parts = []
    for date, group in panel.groupby("date", sort=True):
        g = group.copy()
        for col in CONTINUOUS_FEATURES:
            vals = g[col]
            if vals.notna().sum() >= 5:
                rank = vals.rank(pct=True)
                g[f"x_{col}"] = (rank - 0.5) * 2.0
            else:
                g[f"x_{col}"] = np.nan
        for col in BINARY_FEATURES:
            g[f"x_{col}"] = g[col]
        ranked_parts.append(g)
    panel = pd.concat(ranked_parts, ignore_index=True)
    return panel
