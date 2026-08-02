"""Performance, market-regime, and overfitting diagnostics."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Dict

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .data import adjust_ohlc
from .indicators import ema


def compute_metrics(returns: pd.Series, equity: pd.Series) -> Dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {}
    days = len(returns)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float((1.0 + total_return) ** (252.0 / max(days, 1)) - 1.0)
    ann_vol = float(returns.std() * math.sqrt(252))
    sharpe = float((returns.mean() / returns.std()) * math.sqrt(252)) if returns.std() > 0 else np.nan
    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan
    win_rate = float((returns > 0).mean())
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate_daily": win_rate,
        "days": days,
    }


def compute_market_regime_multiplier(benchmark: pd.DataFrame, dates: pd.DatetimeIndex, cfg: BacktestConfig) -> pd.Series:
    if not cfg.regime_filter:
        return pd.Series(1.0, index=dates)

    bench_df = adjust_ohlc(benchmark).set_index("date").sort_index()
    bench_close = bench_df["adj_close"].astype(float)
    bench_ret = bench_close.pct_change(fill_method=None)
    ema_fast = ema(bench_close, cfg.benchmark_ema_fast)
    ema_slow = ema(bench_close, cfg.benchmark_ema_slow)
    realized_vol = bench_ret.rolling(cfg.benchmark_vol_window).std() * math.sqrt(252)
    vol_pct = realized_vol.rolling(cfg.benchmark_vol_percentile_window).rank(pct=True)
    short_ret = bench_close / bench_close.shift(cfg.benchmark_bad_return_window) - 1.0

    bad = (
        (bench_close < ema_slow)
        | (ema_fast < ema_slow)
        | (vol_pct > cfg.benchmark_vol_percentile_cutoff)
        | (short_ret < cfg.benchmark_bad_return_threshold)
    )
    severe = (
        ((bench_close < ema_slow) & (short_ret < cfg.benchmark_bad_return_threshold))
        | (short_ret < cfg.benchmark_severe_return_threshold)
        | (vol_pct > 0.95)
    )

    multiplier = pd.Series(1.0, index=bench_close.index)
    multiplier.loc[bad.fillna(False)] = cfg.regime_bad_mult
    multiplier.loc[severe.fillna(False)] = cfg.regime_severe_mult
    return multiplier.shift(1).reindex(dates).ffill().fillna(1.0)


def compute_overfit_report(returns: pd.Series, effective_trials: int) -> Dict[str, float]:
    returns = returns.dropna()
    if len(returns) < 3 or returns.std() <= 0:
        return {}

    n = len(returns)
    daily_sr = float(returns.mean() / returns.std())
    skew = float(returns.skew())
    kurt = float(returns.kurtosis() + 3.0)
    denom = 1.0 - skew * daily_sr + ((kurt - 1.0) / 4.0) * daily_sr * daily_sr
    sr_std = math.sqrt(max(denom, 1e-12) / max(n - 1, 1))
    norm = NormalDist()
    k = max(int(effective_trials), 1)
    gamma = 0.5772156649015329
    if k > 1:
        sr_star = sr_std * (
            (1.0 - gamma) * norm.inv_cdf(1.0 - 1.0 / k)
            + gamma * norm.inv_cdf(1.0 - 1.0 / (k * math.e))
        )
    else:
        sr_star = 0.0
    psr = norm.cdf((daily_sr - 0.0) / sr_std)
    dsr_proxy = norm.cdf((daily_sr - sr_star) / sr_std)
    return {
        "effective_trials": float(k),
        "daily_sharpe": daily_sr,
        "annual_sharpe": daily_sr * math.sqrt(252),
        "skew": skew,
        "kurtosis": kurt,
        "single_trial_psr": psr,
        "deflated_sharpe_probability_proxy": dsr_proxy,
        "multiple_testing_daily_sr_threshold": sr_star,
        "multiple_testing_annual_sr_threshold": sr_star * math.sqrt(252),
    }
