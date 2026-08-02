"""Command-line orchestration for data, ML scoring, backtest, and reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from .config import BacktestConfig
from .data import fetch_all_data
from .engine import run_backtest
from .features import compute_features_for_symbol, make_feature_panel
from .model import train_walk_forward_scores
from .reporting import write_summary_markdown
from .universe import fetch_tradingview_universe, load_universe_symbols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VWAP + Volume Profile walk-forward BIST backtest.")
    parser.add_argument("--output-dir", default="outputs/final_all_bist")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--universe-source", choices=["builtin", "file", "tradingview"], default="tradingview")
    parser.add_argument("--universe-file", default=None)
    parser.add_argument("--include-non-normal-instruments", action="store_true")
    parser.add_argument("--data-source", choices=["auto", "yahoo", "tradingview"], default="auto")
    parser.add_argument("--tradingview-data-dir", default=None)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--cost", type=float, default=0.0025)
    parser.add_argument("--vwap-execution-model", dest="use_vwap_execution_model", action="store_true", default=True)
    parser.add_argument("--no-vwap-execution-model", dest="use_vwap_execution_model", action="store_false")
    parser.add_argument("--portfolio-value-tl", type=float, default=10_000_000.0)
    parser.add_argument("--adv-window", type=int, default=20)
    parser.add_argument("--participation-rate", type=float, default=0.03)
    parser.add_argument("--max-liquidity-cap-weight", type=float, default=0.35)
    parser.add_argument("--dynamic-slippage", dest="dynamic_slippage", action="store_true", default=True)
    parser.add_argument("--no-dynamic-slippage", dest="dynamic_slippage", action="store_false")
    parser.add_argument("--base-slippage-bps", type=float, default=3.0)
    parser.add_argument("--participation-slippage-bps", type=float, default=18.0)
    parser.add_argument("--vol-slippage-bps", type=float, default=12.0)
    parser.add_argument("--vwap-chase-slippage-bps", type=float, default=6.0)
    parser.add_argument("--profile-width-slippage-bps", type=float, default=2.0)
    parser.add_argument("--max-dynamic-slippage-bps", type=float, default=150.0)
    parser.add_argument("--completion-sleeve", dest="use_completion_sleeve", action="store_true", default=True)
    parser.add_argument("--no-completion-sleeve", dest="use_completion_sleeve", action="store_false")
    parser.add_argument("--completion-target-weight", type=float, default=1.0)
    parser.add_argument("--completion-top-n", type=int, default=8)
    parser.add_argument("--completion-candidate-pool", type=int, default=40)
    parser.add_argument("--completion-max-position-weight", type=float, default=0.12)
    parser.add_argument("--completion-min-cash-to-fill", type=float, default=0.03)
    parser.add_argument("--completion-liquidity-weight", type=float, default=0.55)
    parser.add_argument("--completion-alpha-weight", type=float, default=0.30)
    parser.add_argument("--completion-low-vol-weight", type=float, default=0.15)
    parser.add_argument("--completion-max-pairwise-corr", type=float, default=0.90)
    parser.add_argument("--factor-beta-window", type=int, default=120)
    parser.add_argument("--rebalance-days", type=int, default=30)
    parser.add_argument("--sell-rank", type=int, default=25)
    parser.add_argument("--weighting", choices=["equal", "score", "invvol", "ml_dynamic"], default="ml_dynamic")
    parser.add_argument("--ml-weight-power", type=float, default=0.05)
    parser.add_argument("--max-position-weight", type=float, default=0.35)
    parser.add_argument("--no-cost-adjust-score", action="store_true")
    parser.add_argument("--liquidity-cost-penalty", type=float, default=0.0)
    parser.add_argument("--vwap-chase-penalty", type=float, default=0.0)
    parser.add_argument("--no-correlation-filter", action="store_true")
    parser.add_argument("--max-pairwise-corr", type=float, default=0.80)
    parser.add_argument("--corr-window", type=int, default=60)
    parser.add_argument("--vol-target", type=float, default=0.38)
    parser.add_argument("--vol-target-window", type=int, default=40)
    parser.add_argument("--min-exposure", type=float, default=0.25)
    parser.add_argument("--max-exposure", type=float, default=1.15)
    parser.add_argument("--drawdown-governor", dest="drawdown_governor", action="store_true", default=True)
    parser.add_argument("--no-drawdown-governor", dest="drawdown_governor", action="store_false")
    parser.add_argument("--dd-level1", type=float, default=-0.12)
    parser.add_argument("--dd-level2", type=float, default=-0.25)
    parser.add_argument("--dd-mult1", type=float, default=0.60)
    parser.add_argument("--dd-mult2", type=float, default=0.35)
    parser.add_argument("--use-meta-label", dest="use_meta_label", action="store_true", default=False)
    parser.add_argument("--no-meta-label", dest="use_meta_label", action="store_false")
    parser.add_argument("--meta-prob-threshold", type=float, default=0.52)
    parser.add_argument("--meta-score-power", type=float, default=1.0)
    parser.add_argument("--meta-profit-atr", type=float, default=2.0)
    parser.add_argument("--meta-stop-atr", type=float, default=1.0)
    parser.add_argument("--regime-filter", dest="regime_filter", action="store_true", default=False)
    parser.add_argument("--no-regime-filter", dest="regime_filter", action="store_false")
    parser.add_argument("--benchmark-vol-percentile-cutoff", type=float, default=0.90)
    parser.add_argument("--benchmark-bad-return-threshold", type=float, default=-0.08)
    parser.add_argument("--benchmark-severe-return-threshold", type=float, default=-0.10)
    parser.add_argument("--regime-bad-mult", type=float, default=0.80)
    parser.add_argument("--regime-severe-mult", type=float, default=0.55)
    parser.add_argument("--event-exit", dest="event_exit", action="store_true", default=False)
    parser.add_argument("--no-event-exit", dest="event_exit", action="store_false")
    parser.add_argument("--momentum-decay-exit", dest="use_momentum_decay_exit", action="store_true", default=True)
    parser.add_argument("--no-momentum-decay-exit", dest="use_momentum_decay_exit", action="store_false")
    parser.add_argument("--decay-reduce-threshold", type=float, default=9.0)
    parser.add_argument("--decay-exit-threshold", type=float, default=1.15)
    parser.add_argument("--decay-reduce-multiplier", type=float, default=1.0)
    parser.add_argument("--decay-rank-threshold", type=int, default=25)
    parser.add_argument("--decay-rank-weight", type=float, default=0.25)
    parser.add_argument("--decay-below-vwap20-weight", type=float, default=0.18)
    parser.add_argument("--decay-below-vwap63-weight", type=float, default=0.12)
    parser.add_argument("--decay-below-poc-weight", type=float, default=0.15)
    parser.add_argument("--decay-ema-break-weight", type=float, default=0.15)
    parser.add_argument("--decay-negative-momentum-weight", type=float, default=0.15)
    parser.add_argument("--decay-failed-vah-weight", type=float, default=0.20)
    parser.add_argument("--effective-trials", type=int, default=288)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BacktestConfig(
        output_dir=Path(args.output_dir),
        refresh_data=args.refresh_data,
        top_n=args.top_n,
        cost_per_turnover=args.cost,
        use_vwap_execution_model=args.use_vwap_execution_model,
        portfolio_value_tl=args.portfolio_value_tl,
        adv_window=args.adv_window,
        participation_rate=args.participation_rate,
        max_liquidity_cap_weight=args.max_liquidity_cap_weight,
        dynamic_slippage=args.dynamic_slippage,
        base_slippage_bps=args.base_slippage_bps,
        participation_slippage_bps=args.participation_slippage_bps,
        vol_slippage_bps=args.vol_slippage_bps,
        vwap_chase_slippage_bps=args.vwap_chase_slippage_bps,
        profile_width_slippage_bps=args.profile_width_slippage_bps,
        max_dynamic_slippage_bps=args.max_dynamic_slippage_bps,
        use_completion_sleeve=args.use_completion_sleeve,
        completion_target_weight=args.completion_target_weight,
        completion_top_n=args.completion_top_n,
        completion_candidate_pool=args.completion_candidate_pool,
        completion_max_position_weight=args.completion_max_position_weight,
        completion_min_cash_to_fill=args.completion_min_cash_to_fill,
        completion_liquidity_weight=args.completion_liquidity_weight,
        completion_alpha_weight=args.completion_alpha_weight,
        completion_low_vol_weight=args.completion_low_vol_weight,
        completion_max_pairwise_corr=args.completion_max_pairwise_corr,
        factor_beta_window=args.factor_beta_window,
        rebalance_every_n_days=args.rebalance_days,
        sell_rank_threshold=args.sell_rank,
        weighting=args.weighting,
        ml_weight_power=args.ml_weight_power,
        max_position_weight=args.max_position_weight,
        cost_adjust_score=not args.no_cost_adjust_score,
        liquidity_cost_penalty=args.liquidity_cost_penalty,
        vwap_chase_penalty=args.vwap_chase_penalty,
        correlation_filter=not args.no_correlation_filter,
        max_pairwise_corr=args.max_pairwise_corr,
        corr_window=args.corr_window,
        vol_target=args.vol_target,
        vol_target_window=args.vol_target_window,
        min_exposure=args.min_exposure,
        max_exposure=args.max_exposure,
        drawdown_governor=args.drawdown_governor,
        dd_level1=args.dd_level1,
        dd_level2=args.dd_level2,
        dd_mult1=args.dd_mult1,
        dd_mult2=args.dd_mult2,
        use_meta_label=args.use_meta_label,
        meta_prob_threshold=args.meta_prob_threshold,
        meta_score_power=args.meta_score_power,
        meta_profit_atr=args.meta_profit_atr,
        meta_stop_atr=args.meta_stop_atr,
        regime_filter=args.regime_filter,
        benchmark_vol_percentile_cutoff=args.benchmark_vol_percentile_cutoff,
        benchmark_bad_return_threshold=args.benchmark_bad_return_threshold,
        benchmark_severe_return_threshold=args.benchmark_severe_return_threshold,
        regime_bad_mult=args.regime_bad_mult,
        regime_severe_mult=args.regime_severe_mult,
        event_exit=args.event_exit,
        use_momentum_decay_exit=args.use_momentum_decay_exit,
        decay_reduce_threshold=args.decay_reduce_threshold,
        decay_exit_threshold=args.decay_exit_threshold,
        decay_reduce_multiplier=args.decay_reduce_multiplier,
        decay_rank_threshold=args.decay_rank_threshold,
        decay_rank_weight=args.decay_rank_weight,
        decay_below_vwap20_weight=args.decay_below_vwap20_weight,
        decay_below_vwap63_weight=args.decay_below_vwap63_weight,
        decay_below_poc_weight=args.decay_below_poc_weight,
        decay_ema_break_weight=args.decay_ema_break_weight,
        decay_negative_momentum_weight=args.decay_negative_momentum_weight,
        decay_failed_vah_weight=args.decay_failed_vah_weight,
        effective_trials=args.effective_trials,
        universe_source=args.universe_source,
        universe_file=Path(args.universe_file) if args.universe_file else None,
        normal_share_only=not args.include_non_normal_instruments,
        data_source=args.data_source,
        tradingview_data_dir=Path(args.tradingview_data_dir) if args.tradingview_data_dir else None,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.universe_file is not None and cfg.universe_source == "builtin":
        cfg.universe_source = "file"
    universe_symbols = load_universe_symbols(cfg)
    symbols = sorted(set(universe_symbols + [cfg.benchmark]))
    pd.DataFrame({"symbol": universe_symbols}).to_csv(cfg.output_dir / "universe_resolved.csv", index=False)
    print(
        f"[UNIVERSE] symbols={len(universe_symbols)} normal_share_only={cfg.normal_share_only} "
        f"source={cfg.universe_source if cfg.universe_file is None else cfg.universe_file}"
    )
    data = fetch_all_data(symbols, cfg)
    if cfg.benchmark not in data:
        raise RuntimeError(f"Benchmark {cfg.benchmark} could not be fetched.")
    benchmark = data[cfg.benchmark]

    feature_map: Dict[str, pd.DataFrame] = {}
    for symbol, df in sorted(data.items()):
        if symbol == cfg.benchmark:
            continue
        try:
            feat = compute_features_for_symbol(df, benchmark, cfg)
            if feat["close"].notna().sum() > 500:
                feature_map[symbol] = feat
                print(f"[FEATURE] {symbol}: {len(feat)} rows")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] feature failed for {symbol}: {exc}")

    if not feature_map:
        raise RuntimeError("No features were computed.")

    panel = make_feature_panel(feature_map)
    panel.to_csv(cfg.output_dir / "feature_panel.csv.gz", index=False, compression="gzip")
    scored_panel, coef_df = train_walk_forward_scores(panel, cfg)
    scored_panel.to_csv(cfg.output_dir / "scored_panel.csv.gz", index=False, compression="gzip")
    coef_df.to_csv(cfg.output_dir / "ml_feature_weights_by_month.csv", index=False)

    equity_curve, rebalances, summary = run_backtest(scored_panel, benchmark, cfg)
    event_exit_records = summary.pop("_event_exit_records", [])
    source_counts = {
        str(source): int(count)
        for source, count in pd.Series(
            [df["data_source"].iloc[0] if "data_source" in df.columns and not df.empty else "unknown" for df in data.values()]
        ).value_counts().items()
    }
    summary["universe"] = {
        "requested_symbols": len(universe_symbols),
        "loaded_symbols_including_benchmark": len(data),
        "feature_symbols": len(feature_map),
        "normal_share_only": cfg.normal_share_only,
        "universe_file": str(cfg.universe_file) if cfg.universe_file else None,
        "universe_source": cfg.universe_source,
        "data_source_mode": cfg.data_source,
        "data_source_counts": source_counts,
    }
    pd.DataFrame(event_exit_records).to_csv(cfg.output_dir / "event_exits.csv", index=False)
    equity_curve.to_csv(cfg.output_dir / "equity_curve.csv")
    rebalances.to_csv(cfg.output_dir / "rebalances.csv", index=False)
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_markdown(summary, coef_df, cfg)

    print(json.dumps(summary, indent=2))
    print(f"[DONE] Outputs written to {cfg.output_dir.resolve()}")


if __name__ == "__main__":
    main()
