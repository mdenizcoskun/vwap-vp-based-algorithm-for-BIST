"""Human-readable backtest report generation."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, BacktestConfig


def write_summary_markdown(summary: Dict[str, object], coef_df: pd.DataFrame, cfg: BacktestConfig) -> None:
    summary_path = cfg.output_dir / "backtest_summary.md"
    lines = []
    lines.append("# VWAP + Volume Profile ML Backtest Summary")
    lines.append("")
    lines.append("## Data Note")
    lines.append("")
    lines.append("- No minimum-volume filter was applied.")
    lines.append("- Data source: Yahoo Finance public chart API or supplied TradingView export, daily OHLCV.")
    lines.append("- True intraday VWAP/Volume Profile could not be tested over five years with this free daily dataset.")
    lines.append("- VWAP is approximated with rolling daily HLC3 * volume.")
    lines.append("- Volume Profile is approximated with a rolling 63-day HLC3 price-bin histogram.")
    lines.append("- ML target is risk-adjusted: forward excess return divided by forward realized volatility.")
    lines.append("- Signals are formed after the close and executed at the next available open; the entry overnight gap is not credited.")
    lines.append("- VWAP execution model caps position size by ADV participation and charges dynamic slippage from participation, volatility, VWAP chase, and profile width.")
    lines.append(f"- Liquid completion sleeve enabled: {cfg.use_completion_sleeve}.")
    lines.append(f"- Execution capacity is estimated on a fixed {cfg.portfolio_value_tl:,.0f} TL simulated book size.")
    lines.append("- Between rebalance signals, positions drift with market moves; the portfolio is not reset to target weights daily except for exposure scaling.")
    lines.append(f"- Portfolio weighting mode: {cfg.weighting}.")
    lines.append(f"- Correlation filter enabled: {cfg.correlation_filter}, max pairwise corr: {cfg.max_pairwise_corr}.")
    lines.append(f"- Portfolio volatility target: {cfg.vol_target if cfg.vol_target > 0 else 'disabled'}.")
    lines.append(f"- Drawdown governor enabled: {cfg.drawdown_governor}.")
    lines.append(f"- Market regime filter enabled: {cfg.regime_filter}.")
    lines.append(f"- Meta-label confirmation enabled: {cfg.use_meta_label}, threshold: {cfg.meta_prob_threshold}.")
    lines.append(f"- Event-driven exits enabled: {cfg.event_exit}.")
    lines.append(f"- Momentum decay exit enabled: {cfg.use_momentum_decay_exit}.")
    lines.append(f"- Deflated Sharpe proxy effective trials: {cfg.effective_trials}.")
    lines.append("- Turnover is reduced with a buy/hold band: buy top-N, keep current names until sell-rank breach.")
    universe = summary.get("universe", {})
    if universe:
        lines.append("")
        lines.append("## Universe")
        lines.append("")
        lines.append(f"- requested_symbols: {universe.get('requested_symbols', 0)}")
        lines.append(f"- loaded_symbols_including_benchmark: {universe.get('loaded_symbols_including_benchmark', 0)}")
        lines.append(f"- feature_symbols: {universe.get('feature_symbols', 0)}")
        lines.append(f"- normal_share_only: {universe.get('normal_share_only', True)}")
        lines.append(f"- universe_source: {universe.get('universe_source')}")
        lines.append(f"- universe_file: {universe.get('universe_file')}")
        lines.append(f"- data_source_mode: {universe.get('data_source_mode')}")
        lines.append(f"- data_source_counts: {universe.get('data_source_counts')}")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    for name in ["strategy", "benchmark"]:
        metrics = summary.get(name, {})
        lines.append(f"### {name.title()}")
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")
        lines.append("")
    lines.append(f"- avg_active_positions: {summary.get('avg_active_positions', np.nan):.2f}")
    lines.append(f"- avg_daily_turnover: {summary.get('avg_daily_turnover', np.nan):.4f}")
    lines.append(f"- avg_transaction_cost_rate: {summary.get('avg_transaction_cost_rate', np.nan):.6f}")
    lines.append(f"- avg_dynamic_slippage_rate: {summary.get('avg_dynamic_slippage_rate', np.nan):.6f}")
    lines.append(f"- avg_trade_participation: {summary.get('avg_trade_participation', np.nan):.4f}")
    lines.append(f"- max_trade_participation: {summary.get('max_trade_participation', np.nan):.4f}")
    lines.append(f"- avg_completion_weight_sum: {summary.get('avg_completion_weight_sum', np.nan):.4f}")
    lines.append(f"- completion_rebalance_count: {summary.get('completion_rebalance_count', 0)}")
    lines.append(f"- avg_exposure: {summary.get('avg_exposure', np.nan):.4f}")
    lines.append(f"- avg_gross_exposure: {summary.get('avg_gross_exposure', np.nan):.4f}")
    lines.append(f"- avg_portfolio_beta: {summary.get('avg_portfolio_beta', np.nan):.4f}")
    lines.append(f"- avg_portfolio_liquidity_score: {summary.get('avg_portfolio_liquidity_score', np.nan):.4f}")
    lines.append(f"- avg_portfolio_momentum_score: {summary.get('avg_portfolio_momentum_score', np.nan):.4f}")
    lines.append(f"- avg_regime_multiplier: {summary.get('avg_regime_multiplier', np.nan):.4f}")
    lines.append(f"- risk_off_days: {summary.get('risk_off_days', 0)}")
    lines.append(f"- n_rebalances: {summary.get('n_rebalances', 0)}")
    lines.append(f"- n_event_exits: {summary.get('n_event_exits', 0)}")
    if summary.get("exit_action_counts"):
        lines.append(f"- exit_action_counts: {summary.get('exit_action_counts')}")
    overfit = summary.get("overfit_control", {})
    if overfit:
        lines.append("")
        lines.append("## Overfit / False Positive Control")
        lines.append("")
        for key, value in overfit.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.6f}")
            else:
                lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Average ML Feature Weights")
    lines.append("")
    if not coef_df.empty:
        feature_cols = [c for c in coef_df.columns if c in FEATURE_COLUMNS]
        avg = coef_df[feature_cols].mean().sort_values(ascending=False)
        lines.append("| Feature | Avg coefficient |")
        lines.append("|---|---:|")
        for feature, coef in avg.items():
            lines.append(f"| {feature} | {coef:.6f} |")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
