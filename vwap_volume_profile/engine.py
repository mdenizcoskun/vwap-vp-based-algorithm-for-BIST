"""Event-driven portfolio simulation and accounting engine."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .data import adjust_ohlc
from .metrics import compute_market_regime_multiplier, compute_metrics, compute_overfit_report
from .portfolio import (
    apply_vwap_liquidity_caps,
    build_completion_sleeve,
    compute_liquidity_weight_caps,
    compute_portfolio_weights,
    estimate_vwap_execution_cost_rate,
    optional_pivot,
    select_low_correlation_names,
    weighted_factor_exposure,
)


def run_backtest(panel: pd.DataFrame, benchmark: pd.DataFrame, cfg: BacktestConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    required_columns = {"date", "symbol", "open", "close", "ml_score"}
    missing_columns = required_columns - set(panel.columns)
    if missing_columns:
        raise ValueError(f"Panel is missing required columns: {sorted(missing_columns)}")

    open_prices = panel.pivot(index="date", columns="symbol", values="open").sort_index()
    close = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    score = panel.pivot(index="date", columns="symbol", values="ml_score").sort_index()
    if "meta_prob" in panel.columns:
        meta_prob = panel.pivot(index="date", columns="symbol", values="meta_prob").reindex_like(score)
    else:
        meta_prob = pd.DataFrame(1.0, index=score.index, columns=score.columns)
    failed = panel.pivot(index="date", columns="symbol", values="failed_vah_breakout").reindex_like(score)
    dist_vwap = panel.pivot(index="date", columns="symbol", values="dist_vwap20_atr").reindex_like(score)
    above_vwap = panel.pivot(index="date", columns="symbol", values="close_above_vwap20").reindex_like(score)
    above_vwap63 = optional_pivot(panel, score, "close_above_vwap63", default=1.0)
    above_poc = panel.pivot(index="date", columns="symbol", values="close_above_poc").reindex_like(score)
    close_above_ema20 = optional_pivot(panel, score, "close_above_ema20", default=1.0)
    ema_stack = optional_pivot(panel, score, "ema_stack", default=1.0)
    rr_proxy = panel.pivot(index="date", columns="symbol", values="profile_width_atr").reindex_like(score)
    liquidity_x = panel.pivot(index="date", columns="symbol", values="x_turnover_rank_feature").reindex_like(score)
    turnover_tl = optional_pivot(panel, score, "turnover_tl")
    adv_window = max(1, int(cfg.adv_window))
    adv_min_periods = max(1, min(adv_window, max(5, adv_window // 2)))
    adv_tl = turnover_tl.rolling(adv_window, min_periods=adv_min_periods).mean()
    atr14 = optional_pivot(panel, score, "atr14")
    atr_pct = (atr14 / close).replace([np.inf, -np.inf], np.nan)
    momentum_x = optional_pivot(panel, score, "x_mom126_21")
    mom63_5 = optional_pivot(panel, score, "mom63_5")
    mom126_21 = optional_pivot(panel, score, "mom126_21")

    dates = close.index[(close.index >= pd.Timestamp(cfg.test_start)) & (close.index <= pd.Timestamp(cfg.end_date))]
    all_returns = close.pct_change(fill_method=None)
    bench_df = adjust_ohlc(benchmark).set_index("date").sort_index()
    bench_close_all = bench_df["adj_close"].reindex(close.index).ffill()
    benchmark_returns_all = bench_close_all.pct_change(fill_method=None)
    beta_window = max(20, int(cfg.factor_beta_window))
    beta_min_periods = max(20, min(beta_window, beta_window // 3))
    bench_var = benchmark_returns_all.rolling(beta_window, min_periods=beta_min_periods).var()
    rolling_beta = all_returns.rolling(beta_window, min_periods=beta_min_periods).cov(benchmark_returns_all)
    rolling_beta = rolling_beta.divide(bench_var, axis=0).replace([np.inf, -np.inf], np.nan)
    overnight_returns_all = open_prices / close.shift(1) - 1.0
    intraday_returns_all = close / open_prices - 1.0
    overnight_returns = overnight_returns_all.reindex(dates)
    intraday_returns = intraday_returns_all.reindex(dates)
    trailing_vol_all = all_returns.rolling(cfg.vol_weight_window).std()
    trailing_vol = trailing_vol_all.reindex(dates)
    execution_adv_tl = adv_tl.shift(1).reindex(dates)
    execution_atr_pct = atr_pct.shift(1).reindex(dates)
    execution_dist_vwap = dist_vwap.shift(1).reindex(dates)
    execution_profile_width = rr_proxy.shift(1).reindex(dates)
    execution_beta = rolling_beta.shift(1).reindex(dates)
    execution_liquidity_x = liquidity_x.shift(1).reindex(dates)
    execution_trailing_vol = trailing_vol_all.shift(1).reindex(dates)
    execution_momentum_x = momentum_x.shift(1).reindex(dates)
    regime_multiplier = compute_market_regime_multiplier(benchmark, dates, cfg)
    target_weights = pd.DataFrame(0.0, index=dates, columns=close.columns)
    rebalance_dates = dates[:: cfg.rebalance_every_n_days]
    rebalance_log = []
    exit_log = []
    last_weights = pd.Series(0.0, index=close.columns)

    for date in dates:
        if date in rebalance_dates:
            s = score.loc[date].dropna()
            if not s.empty:
                eligible = pd.Series(True, index=s.index)
                eligible &= failed.loc[date, s.index].fillna(1.0) < 0.5
                eligible &= dist_vwap.loc[date, s.index].fillna(999.0) < 3.0
                eligible &= above_vwap.loc[date, s.index].fillna(0.0).gt(0.5) | above_poc.loc[date, s.index].fillna(0.0).gt(0.5)
                eligible &= rr_proxy.loc[date, s.index].fillna(0.0) > 0.10
                if cfg.use_meta_label:
                    mp = meta_prob.loc[date, s.index].fillna(0.0).clip(0.0, 1.0)
                    eligible &= mp >= cfg.meta_prob_threshold
                    s = s * mp.pow(cfg.meta_score_power)
                if cfg.cost_adjust_score:
                    liq_pct = ((liquidity_x.loc[date, s.index].fillna(0.0) + 1.0) / 2.0).clip(0.0, 1.0)
                    chase = dist_vwap.loc[date, s.index].clip(lower=0.0, upper=3.0).fillna(3.0) / 3.0
                    s = s - cfg.liquidity_cost_penalty * (1.0 - liq_pct) - cfg.vwap_chase_penalty * chase
                eligible &= s > 0.0
                ranked = s[eligible].sort_values(ascending=False)
                rank_map = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)
                current_holdings = last_weights[last_weights > 0].index
                keep = [
                    symbol
                    for symbol in current_holdings
                    if symbol in rank_map.index and rank_map.loc[symbol] <= cfg.sell_rank_threshold
                ]
                corr_matrix = None
                if cfg.correlation_filter or cfg.weighting == "ml_dynamic":
                    corr_window = all_returns.loc[:date].tail(cfg.corr_window)
                    corr_matrix = corr_window.corr(min_periods=max(10, cfg.corr_window // 3))
                if len(keep) > cfg.top_n:
                    keep = ranked.loc[keep].sort_values(ascending=False).head(cfg.top_n).index.tolist()
                if cfg.correlation_filter:
                    selected_symbols = select_low_correlation_names(
                        ranked=ranked,
                        keep=keep,
                        top_n=cfg.top_n,
                        corr_matrix=corr_matrix,
                        max_pairwise_corr=cfg.max_pairwise_corr,
                    )
                else:
                    additions = [
                        symbol
                        for symbol in ranked.index
                        if symbol not in keep
                    ][: max(cfg.top_n - len(keep), 0)]
                    selected_symbols = keep + additions
                selected = ranked.loc[selected_symbols].sort_values(ascending=False)
                new_weights = pd.Series(0.0, index=close.columns)
                alpha_weight_sum = 0.0
                completion_weight_sum = 0.0
                completion = pd.Series(dtype=float)
                if len(selected) > 0:
                    weights = compute_portfolio_weights(
                        selected=selected,
                        trailing_vol_row=trailing_vol.loc[date],
                        corr_matrix=corr_matrix,
                        cfg=cfg,
                    )
                    weights = apply_vwap_liquidity_caps(weights, adv_tl.loc[date], cfg)
                    alpha_weight_sum = float(weights.sum())
                    completion = build_completion_sleeve(
                        alpha_weights=weights,
                        ranked=ranked,
                        trailing_vol_row=trailing_vol.loc[date],
                        adv_tl_row=adv_tl.loc[date],
                        corr_matrix=corr_matrix,
                        cfg=cfg,
                    )
                    completion_weight_sum = float(completion.sum()) if not completion.empty else 0.0
                    combined_weights = weights.add(completion, fill_value=0.0)
                    new_weights.loc[combined_weights.index] = combined_weights
                last_weights = new_weights
                rebalance_log.append(
                    {
                        "date": str(pd.Timestamp(date).date()),
                        "n_selected": int((new_weights > 0).sum()),
                        "n_alpha_selected": int(len(selected)),
                        "n_completion": int((completion > 0).sum()) if not completion.empty else 0,
                        "symbols": ",".join(new_weights[new_weights > 0].index.tolist()),
                        "weights": ",".join(f"{symbol}:{weight:.4f}" for symbol, weight in new_weights[new_weights > 0].items()),
                        "target_weight_sum": float(new_weights.sum()),
                        "alpha_weight_sum": alpha_weight_sum,
                        "completion_weight_sum": completion_weight_sum,
                        "completion_symbols": ",".join(completion[completion > 0].index.tolist()) if not completion.empty else "",
                        "avg_score": float(selected.mean()) if len(selected) else np.nan,
                        "avg_meta_prob": float(meta_prob.loc[date, selected.index].mean()) if len(selected) else np.nan,
                    }
                )
        elif (cfg.event_exit or cfg.use_momentum_decay_exit) and (last_weights > 0).any():
            held = last_weights[last_weights > 0].index
            exit_reasons: Dict[str, List[str]] = {symbol: [] for symbol in held}
            full_exits: set[str] = set()
            reduce_symbols: Dict[str, Tuple[float, List[str]]] = {}

            if cfg.event_exit and cfg.exit_on_failed_vah:
                for symbol in held[failed.loc[date, held].fillna(0.0).gt(0.5)]:
                    exit_reasons[symbol].append("failed_vah")
            if cfg.event_exit and cfg.exit_on_below_vwap20:
                for symbol in held[above_vwap.loc[date, held].fillna(1.0).le(0.5)]:
                    exit_reasons[symbol].append("below_vwap20")
            if cfg.event_exit and cfg.exit_on_below_poc:
                for symbol in held[above_poc.loc[date, held].fillna(1.0).le(0.5)]:
                    exit_reasons[symbol].append("below_poc")
            if cfg.event_exit and cfg.exit_on_nonpositive_score:
                score_row = score.loc[date, held].fillna(-np.inf)
                for symbol in held[score_row.le(0.0)]:
                    exit_reasons[symbol].append("nonpositive_score")
            if cfg.event_exit and cfg.use_meta_label:
                mp_row = meta_prob.loc[date, held].fillna(1.0)
                for symbol in held[mp_row.lt(cfg.meta_prob_threshold)]:
                    exit_reasons[symbol].append("meta_prob")
            full_exits.update([symbol for symbol, reasons in exit_reasons.items() if reasons])

            if cfg.use_momentum_decay_exit:
                score_row = score.loc[date].replace([np.inf, -np.inf], np.nan).dropna()
                ranked_all = score_row[score_row > 0.0].sort_values(ascending=False)
                rank_map = pd.Series(np.arange(1, len(ranked_all) + 1), index=ranked_all.index)
                rank_threshold = cfg.decay_rank_threshold if cfg.decay_rank_threshold > 0 else cfg.sell_rank_threshold

                for symbol in held:
                    decay_score = 0.0
                    reasons: List[str] = []
                    rank_value = rank_map.get(symbol, np.inf)
                    if not np.isfinite(rank_value) or rank_value > rank_threshold:
                        decay_score += cfg.decay_rank_weight
                        reasons.append("rank_decay")
                    if above_vwap.loc[date, symbol] <= 0.5 if symbol in above_vwap.columns else False:
                        decay_score += cfg.decay_below_vwap20_weight
                        reasons.append("below_vwap20")
                    if above_vwap63.loc[date, symbol] <= 0.5 if symbol in above_vwap63.columns else False:
                        decay_score += cfg.decay_below_vwap63_weight
                        reasons.append("below_vwap63")
                    if above_poc.loc[date, symbol] <= 0.5 if symbol in above_poc.columns else False:
                        decay_score += cfg.decay_below_poc_weight
                        reasons.append("below_poc")
                    ema_broken = False
                    if symbol in close_above_ema20.columns:
                        ema_broken = ema_broken or bool(close_above_ema20.loc[date, symbol] <= 0.5)
                    if symbol in ema_stack.columns:
                        ema_broken = ema_broken or bool(ema_stack.loc[date, symbol] <= 0.5)
                    if ema_broken:
                        decay_score += cfg.decay_ema_break_weight
                        reasons.append("ema_break")
                    negative_momentum = False
                    if symbol in mom63_5.columns:
                        negative_momentum = negative_momentum or bool(mom63_5.loc[date, symbol] < 0.0)
                    if symbol in mom126_21.columns:
                        negative_momentum = negative_momentum or bool(mom126_21.loc[date, symbol] < 0.0)
                    if negative_momentum:
                        decay_score += cfg.decay_negative_momentum_weight
                        reasons.append("negative_momentum")
                    if symbol in failed.columns and failed.loc[date, symbol] > 0.5:
                        decay_score += cfg.decay_failed_vah_weight
                        reasons.append("failed_vah")

                    if decay_score >= cfg.decay_exit_threshold:
                        exit_reasons[symbol].extend([f"decay:{reason}" for reason in reasons])
                        full_exits.add(symbol)
                    elif decay_score >= cfg.decay_reduce_threshold:
                        reduce_symbols[symbol] = (decay_score, reasons)

            for symbol in sorted(full_exits):
                previous_weight = float(last_weights.loc[symbol])
                exit_log.append(
                    {
                        "date": str(pd.Timestamp(date).date()),
                        "symbol": symbol,
                        "action": "exit",
                        "reasons": ",".join(exit_reasons.get(symbol, [])),
                        "decay_score": np.nan,
                        "previous_weight": previous_weight,
                        "new_weight": 0.0,
                    }
                )
                last_weights.loc[symbol] = 0.0

            for symbol, (decay_score, reasons) in reduce_symbols.items():
                if symbol in full_exits or last_weights.loc[symbol] <= 0:
                    continue
                previous_weight = float(last_weights.loc[symbol])
                new_weight = previous_weight * float(np.clip(cfg.decay_reduce_multiplier, 0.0, 1.0))
                if new_weight < previous_weight - 1e-12:
                    exit_log.append(
                        {
                            "date": str(pd.Timestamp(date).date()),
                            "symbol": symbol,
                            "action": "reduce",
                            "reasons": ",".join([f"decay:{reason}" for reason in reasons]),
                            "decay_score": float(decay_score),
                            "previous_weight": previous_weight,
                            "new_weight": new_weight,
                        }
                    )
                    last_weights.loc[symbol] = new_weight
        target_weights.loc[date] = last_weights

    base_effective_weights = target_weights.shift(1).fillna(0.0)
    applied_weights = pd.DataFrame(0.0, index=dates, columns=close.columns)
    gross_returns = pd.Series(0.0, index=dates)
    turnover = pd.Series(0.0, index=dates)
    transaction_cost_rate = pd.Series(0.0, index=dates)
    dynamic_slippage_rate = pd.Series(0.0, index=dates)
    avg_trade_participation = pd.Series(0.0, index=dates)
    max_trade_participation = pd.Series(0.0, index=dates)
    net_returns = pd.Series(0.0, index=dates)
    exposure = pd.Series(1.0, index=dates)
    gross_exposure = pd.Series(0.0, index=dates)
    portfolio_beta = pd.Series(np.nan, index=dates)
    portfolio_liquidity_score = pd.Series(np.nan, index=dates)
    portfolio_volatility_proxy = pd.Series(np.nan, index=dates)
    portfolio_momentum_score = pd.Series(np.nan, index=dates)
    equity = pd.Series(1.0, index=dates)
    prev_close_weights = pd.Series(0.0, index=close.columns)
    prev_target_composition = pd.Series(0.0, index=close.columns)
    peak_equity = 1.0

    # Event order: prior close weights earn the overnight gap, trades execute at
    # the open, transaction costs are charged, then open-to-close returns accrue.
    for idx, date in enumerate(dates):
        if idx == 0:
            prior_equity = 1.0
        else:
            prior_equity = equity.iloc[idx - 1]
        peak_equity = max(peak_equity, prior_equity)
        drawdown = prior_equity / peak_equity - 1.0

        current_exposure = 1.0
        if cfg.vol_target > 0 and idx > cfg.vol_target_window:
            realized_vol = net_returns.iloc[max(0, idx - cfg.vol_target_window):idx].std() * math.sqrt(252)
            if realized_vol and np.isfinite(realized_vol) and realized_vol > 0:
                current_exposure = cfg.vol_target / realized_vol
                current_exposure = float(np.clip(current_exposure, cfg.min_exposure, cfg.max_exposure))

        if cfg.drawdown_governor:
            if drawdown <= cfg.dd_level2:
                current_exposure *= cfg.dd_mult2
            elif drawdown <= cfg.dd_level1:
                current_exposure *= cfg.dd_mult1
        current_exposure *= float(regime_multiplier.loc[date])

        overnight_ret = overnight_returns.loc[date].fillna(0.0)
        intraday_ret = intraday_returns.loc[date].fillna(0.0)
        overnight_gross = (prev_close_weights * overnight_ret).sum()

        drifted_open_weights = prev_close_weights * (1.0 + overnight_ret)
        if 1.0 + overnight_gross > 0:
            drifted_open_weights = drifted_open_weights / (1.0 + overnight_gross)
        drifted_open_weights = drifted_open_weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        target_composition = base_effective_weights.loc[date].fillna(0.0)
        signal_changed = (target_composition - prev_target_composition).abs().sum() > 1e-10
        target_gross = float(target_composition.sum())
        if signal_changed:
            weights_today = target_composition * current_exposure
        elif drifted_open_weights.abs().sum() > 0 and current_exposure > 0:
            current_gross = float(drifted_open_weights.sum())
            desired_gross = target_gross * current_exposure
            if current_gross > 0:
                weights_today = drifted_open_weights / current_gross * desired_gross
            else:
                weights_today = drifted_open_weights
        elif target_gross > 0:
            weights_today = target_composition * current_exposure
        else:
            weights_today = pd.Series(0.0, index=close.columns)

        applied_weights.loc[date] = weights_today
        exposure.loc[date] = current_exposure
        gross_exposure.loc[date] = weights_today.abs().sum()
        portfolio_beta.loc[date] = weighted_factor_exposure(weights_today, execution_beta.loc[date])
        portfolio_liquidity_score.loc[date] = weighted_factor_exposure(weights_today, execution_liquidity_x.loc[date])
        portfolio_volatility_proxy.loc[date] = weighted_factor_exposure(weights_today, execution_trailing_vol.loc[date])
        portfolio_momentum_score.loc[date] = weighted_factor_exposure(weights_today, execution_momentum_x.loc[date])

        delta_weights = weights_today - drifted_open_weights
        turnover.loc[date] = delta_weights.abs().sum()
        intraday_gross = (weights_today * intraday_ret).sum()
        gross_returns.loc[date] = (1.0 + overnight_gross) * (1.0 + intraday_gross) - 1.0

        equity_after_overnight = prior_equity * (1.0 + overnight_gross)
        cost_stats = estimate_vwap_execution_cost_rate(
            delta_weights=delta_weights,
            adv_tl_row=execution_adv_tl.loc[date],
            atr_pct_row=execution_atr_pct.loc[date],
            dist_vwap_row=execution_dist_vwap.loc[date],
            profile_width_row=execution_profile_width.loc[date],
            cfg=cfg,
            equity_after_overnight=equity_after_overnight,
        )
        transaction_cost_rate.loc[date] = cost_stats["total_cost_rate"]
        dynamic_slippage_rate.loc[date] = cost_stats["dynamic_slippage_rate"]
        avg_trade_participation.loc[date] = cost_stats["avg_trade_participation"]
        max_trade_participation.loc[date] = cost_stats["max_trade_participation"]
        transaction_cost = equity_after_overnight * transaction_cost_rate.loc[date]
        equity_after_cost = equity_after_overnight - transaction_cost
        equity.loc[date] = equity_after_cost * (1.0 + intraday_gross)
        net_returns.loc[date] = equity.loc[date] / prior_equity - 1.0
        close_weights = weights_today * (1.0 + intraday_ret)
        if 1.0 + intraday_gross > 0:
            close_weights = close_weights / (1.0 + intraday_gross)
        prev_close_weights = close_weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        prev_target_composition = target_composition

    bench_df = adjust_ohlc(benchmark).set_index("date").sort_index()
    bench_close = bench_df["adj_close"].reindex(dates).ffill()
    bench_ret = bench_close.pct_change(fill_method=None).fillna(0.0)
    bench_equity = (1.0 + bench_ret).cumprod()
    bench_equity.iloc[0] = 1.0

    equity_curve = pd.DataFrame(
        {
            "strategy_return": net_returns,
            "gross_return": gross_returns,
            "turnover": turnover,
            "transaction_cost_rate": transaction_cost_rate,
            "dynamic_slippage_rate": dynamic_slippage_rate,
            "avg_trade_participation": avg_trade_participation,
            "max_trade_participation": max_trade_participation,
            "equity": equity,
            "benchmark_return": bench_ret,
            "benchmark_equity": bench_equity,
            "active_positions": (applied_weights > 0).sum(axis=1),
            "exposure": exposure,
            "gross_exposure": gross_exposure,
            "portfolio_beta": portfolio_beta,
            "portfolio_liquidity_score": portfolio_liquidity_score,
            "portfolio_volatility_proxy": portfolio_volatility_proxy,
            "portfolio_momentum_score": portfolio_momentum_score,
            "regime_multiplier": regime_multiplier,
        }
    )
    rebalances = pd.DataFrame(rebalance_log)
    exits = pd.DataFrame(exit_log)

    summary = {
        "config": {
            "test_start": cfg.test_start,
            "end_date": cfg.end_date,
            "top_n": cfg.top_n,
            "rebalance_every_n_days": cfg.rebalance_every_n_days,
            "sell_rank_threshold": cfg.sell_rank_threshold,
            "cost_per_turnover": cfg.cost_per_turnover,
            "minimum_volume_filter": None,
            "use_vwap_execution_model": cfg.use_vwap_execution_model,
            "portfolio_value_tl": cfg.portfolio_value_tl,
            "adv_window": cfg.adv_window,
            "participation_rate": cfg.participation_rate,
            "max_liquidity_cap_weight": cfg.max_liquidity_cap_weight,
            "dynamic_slippage": cfg.dynamic_slippage,
            "base_slippage_bps": cfg.base_slippage_bps,
            "participation_slippage_bps": cfg.participation_slippage_bps,
            "vol_slippage_bps": cfg.vol_slippage_bps,
            "vwap_chase_slippage_bps": cfg.vwap_chase_slippage_bps,
            "profile_width_slippage_bps": cfg.profile_width_slippage_bps,
            "max_dynamic_slippage_bps": cfg.max_dynamic_slippage_bps,
            "use_completion_sleeve": cfg.use_completion_sleeve,
            "completion_target_weight": cfg.completion_target_weight,
            "completion_top_n": cfg.completion_top_n,
            "completion_candidate_pool": cfg.completion_candidate_pool,
            "completion_max_position_weight": cfg.completion_max_position_weight,
            "completion_min_cash_to_fill": cfg.completion_min_cash_to_fill,
            "completion_liquidity_weight": cfg.completion_liquidity_weight,
            "completion_alpha_weight": cfg.completion_alpha_weight,
            "completion_low_vol_weight": cfg.completion_low_vol_weight,
            "completion_max_pairwise_corr": cfg.completion_max_pairwise_corr,
            "factor_beta_window": cfg.factor_beta_window,
            "profile_window": cfg.profile_window,
            "profile_bins": cfg.profile_bins,
            "label": "forward excess return divided by forward realized volatility",
            "weighting": cfg.weighting,
            "vol_weight_window": cfg.vol_weight_window,
            "max_position_weight": cfg.max_position_weight,
            "ml_weight_power": cfg.ml_weight_power,
            "cost_adjust_score": cfg.cost_adjust_score,
            "liquidity_cost_penalty": cfg.liquidity_cost_penalty,
            "vwap_chase_penalty": cfg.vwap_chase_penalty,
            "correlation_filter": cfg.correlation_filter,
            "max_pairwise_corr": cfg.max_pairwise_corr,
            "corr_window": cfg.corr_window,
            "vol_target": cfg.vol_target,
            "vol_target_window": cfg.vol_target_window,
            "min_exposure": cfg.min_exposure,
            "max_exposure": cfg.max_exposure,
            "drawdown_governor": cfg.drawdown_governor,
            "use_meta_label": cfg.use_meta_label,
            "meta_prob_threshold": cfg.meta_prob_threshold,
            "meta_score_power": cfg.meta_score_power,
            "regime_filter": cfg.regime_filter,
            "benchmark_vol_percentile_cutoff": cfg.benchmark_vol_percentile_cutoff,
            "benchmark_bad_return_threshold": cfg.benchmark_bad_return_threshold,
            "benchmark_severe_return_threshold": cfg.benchmark_severe_return_threshold,
            "regime_bad_mult": cfg.regime_bad_mult,
            "regime_severe_mult": cfg.regime_severe_mult,
            "event_exit": cfg.event_exit,
            "use_momentum_decay_exit": cfg.use_momentum_decay_exit,
            "decay_reduce_threshold": cfg.decay_reduce_threshold,
            "decay_exit_threshold": cfg.decay_exit_threshold,
            "decay_reduce_multiplier": cfg.decay_reduce_multiplier,
            "decay_rank_threshold": cfg.decay_rank_threshold,
            "decay_rank_weight": cfg.decay_rank_weight,
            "decay_below_vwap20_weight": cfg.decay_below_vwap20_weight,
            "decay_below_vwap63_weight": cfg.decay_below_vwap63_weight,
            "decay_below_poc_weight": cfg.decay_below_poc_weight,
            "decay_ema_break_weight": cfg.decay_ema_break_weight,
            "decay_negative_momentum_weight": cfg.decay_negative_momentum_weight,
            "decay_failed_vah_weight": cfg.decay_failed_vah_weight,
            "effective_trials": cfg.effective_trials,
            "universe_file": str(cfg.universe_file) if cfg.universe_file else None,
            "universe_source": cfg.universe_source,
            "normal_share_only": cfg.normal_share_only,
            "data_source": cfg.data_source,
            "tradingview_data_dir": str(cfg.tradingview_data_dir) if cfg.tradingview_data_dir else None,
        },
        "strategy": compute_metrics(equity_curve["strategy_return"], equity_curve["equity"]),
        "benchmark": compute_metrics(equity_curve["benchmark_return"], equity_curve["benchmark_equity"]),
        "overfit_control": compute_overfit_report(equity_curve["strategy_return"], cfg.effective_trials),
        "avg_active_positions": float(equity_curve["active_positions"].mean()),
        "avg_daily_turnover": float(equity_curve["turnover"].mean()),
        "avg_transaction_cost_rate": float(equity_curve["transaction_cost_rate"].mean()),
        "avg_dynamic_slippage_rate": float(equity_curve["dynamic_slippage_rate"].mean()),
        "avg_trade_participation": float(equity_curve.loc[equity_curve["turnover"] > 0, "avg_trade_participation"].mean()),
        "max_trade_participation": float(equity_curve["max_trade_participation"].max()),
        "avg_exposure": float(equity_curve["exposure"].mean()),
        "avg_gross_exposure": float(equity_curve["gross_exposure"].mean()),
        "avg_portfolio_beta": float(equity_curve["portfolio_beta"].mean()),
        "avg_portfolio_liquidity_score": float(equity_curve["portfolio_liquidity_score"].mean()),
        "avg_portfolio_volatility_proxy": float(equity_curve["portfolio_volatility_proxy"].mean()),
        "avg_portfolio_momentum_score": float(equity_curve["portfolio_momentum_score"].mean()),
        "avg_regime_multiplier": float(equity_curve["regime_multiplier"].mean()),
        "risk_off_days": int((equity_curve["regime_multiplier"] < 1.0).sum()),
        "n_rebalances": int(len(rebalances)),
        "n_event_exits": int(len(exits)),
    }
    if "completion_weight_sum" in rebalances.columns and len(rebalances) > 0:
        summary["avg_completion_weight_sum"] = float(rebalances["completion_weight_sum"].mean())
        summary["completion_rebalance_count"] = int(rebalances["completion_weight_sum"].gt(0).sum())
    else:
        summary["avg_completion_weight_sum"] = 0.0
        summary["completion_rebalance_count"] = 0
    if not exits.empty:
        summary["event_exit_reasons"] = exits["reasons"].str.get_dummies(sep=",").sum().to_dict()
        if "action" in exits.columns:
            summary["exit_action_counts"] = exits["action"].value_counts().to_dict()
    summary["_event_exit_records"] = exit_log
    return equity_curve, rebalances, summary
