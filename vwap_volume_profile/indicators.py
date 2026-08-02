"""Technical indicators, VWAP helpers, and volume-profile calculations."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    tr = true_range(high, low, close)
    tr_sum = tr.rolling(window).sum()
    plus_di = 100 * plus_dm.rolling(window).sum() / tr_sum.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(window).sum() / tr_sum.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(window).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def rolling_beta(stock_ret: pd.Series, bench_ret: pd.Series, window: int = 63) -> pd.Series:
    cov = stock_ret.rolling(window).cov(bench_ret)
    var = bench_ret.rolling(window).var()
    return cov / var.replace(0, np.nan)


def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return numer / denom.replace(0, np.nan)


def compute_volume_profile_levels(
    typical: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    window: int,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(typical)
    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)
    for i in range(window, n):
        lo = np.nanmin(low[i - window : i])
        hi = np.nanmax(high[i - window : i])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        prices = typical[i - window : i]
        vols = np.nan_to_num(volume[i - window : i], nan=0.0)
        idx = np.digitize(prices, bins) - 1
        idx = np.clip(idx, 0, n_bins - 1)
        hist = np.bincount(idx, weights=vols, minlength=n_bins).astype(float)
        if hist.sum() <= 0:
            continue
        centers = (bins[:-1] + bins[1:]) / 2.0
        poc_idx = int(np.argmax(hist))
        target = hist.sum() * 0.70
        included = {poc_idx}
        total = hist[poc_idx]
        left = poc_idx - 1
        right = poc_idx + 1
        while total < target and (left >= 0 or right < n_bins):
            left_vol = hist[left] if left >= 0 else -1
            right_vol = hist[right] if right < n_bins else -1
            if right_vol >= left_vol:
                included.add(right)
                total += max(right_vol, 0)
                right += 1
            else:
                included.add(left)
                total += max(left_vol, 0)
                left -= 1
        poc[i] = centers[poc_idx]
        val[i] = centers[min(included)]
        vah[i] = centers[max(included)]
    return poc, vah, val


def triple_barrier_meta_label(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    horizon: int,
    profit_atr: float,
    stop_atr: float,
) -> pd.Series:
    labels = np.full(len(close), np.nan)
    close_arr = close.to_numpy(dtype=float)
    high_arr = high.to_numpy(dtype=float)
    low_arr = low.to_numpy(dtype=float)
    atr_arr = atr.to_numpy(dtype=float)

    for i in range(len(close_arr) - horizon):
        if not np.isfinite(close_arr[i]) or not np.isfinite(atr_arr[i]) or atr_arr[i] <= 0:
            continue
        profit_level = close_arr[i] + profit_atr * atr_arr[i]
        stop_level = close_arr[i] - stop_atr * atr_arr[i]
        outcome = np.nan
        for j in range(i + 1, min(i + horizon + 1, len(close_arr))):
            hit_stop = np.isfinite(low_arr[j]) and low_arr[j] <= stop_level
            hit_profit = np.isfinite(high_arr[j]) and high_arr[j] >= profit_level
            if hit_stop and hit_profit:
                outcome = 0.0
                break
            if hit_profit:
                outcome = 1.0
                break
            if hit_stop:
                outcome = 0.0
                break
        if np.isnan(outcome) and np.isfinite(close_arr[i + horizon]):
            outcome = 1.0 if close_arr[i + horizon] > close_arr[i] else 0.0
        labels[i] = outcome

    return pd.Series(labels, index=close.index)
