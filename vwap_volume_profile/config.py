"""Configuration and feature definitions for the backtest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


BIST_UNIVERSE = [
    "AEFES.IS", "AGHOL.IS", "AHGAZ.IS", "AKBNK.IS", "AKCNS.IS", "AKFGY.IS",
    "AKFYE.IS", "AKSA.IS", "AKSEN.IS", "ALARK.IS", "ALBRK.IS", "ALFAS.IS",
    "ANHYT.IS", "ANSGR.IS", "ARCLK.IS", "ARDYZ.IS", "ASELS.IS", "ASTOR.IS",
    "AYDEM.IS", "BERA.IS", "BIMAS.IS", "BOBET.IS", "BRSAN.IS", "BRYAT.IS",
    "BTCIM.IS", "CANTE.IS", "CCOLA.IS", "CIMSA.IS", "CLEBI.IS", "CWENE.IS",
    "DOAS.IS", "DOHOL.IS", "ECILC.IS", "ECZYT.IS", "EGEEN.IS", "EKGYO.IS",
    "ENERY.IS", "ENJSA.IS", "ENKAI.IS", "EREGL.IS", "EUPWR.IS", "FROTO.IS",
    "GARAN.IS", "GESAN.IS", "GLYHO.IS", "GOLTS.IS", "GUBRF.IS", "HALKB.IS",
    "HEKTS.IS", "IPEKE.IS", "ISCTR.IS", "ISDMR.IS", "ISGYO.IS", "KARSN.IS",
    "KCAER.IS", "KCHOL.IS", "KLSER.IS", "KMPUR.IS", "KONTR.IS", "KONYA.IS",
    "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "MAVI.IS", "MGROS.IS", "MIATK.IS",
    "MPARK.IS", "ODAS.IS", "OTKAR.IS", "OYAKC.IS", "PASEU.IS", "PETKM.IS",
    "PGSUS.IS", "QUAGR.IS", "REEDR.IS", "SAHOL.IS", "SASA.IS", "SDTTR.IS",
    "SISE.IS", "SKBNK.IS", "SMRTG.IS", "SOKM.IS", "TABGD.IS", "TAVHL.IS",
    "TCELL.IS", "THYAO.IS", "TKFEN.IS", "TOASO.IS", "TSKB.IS", "TTKOM.IS",
    "TTRAK.IS", "TUKAS.IS", "TUPRS.IS", "TURSG.IS", "ULKER.IS", "VAKBN.IS",
    "VESTL.IS", "YEOTK.IS", "YKBNK.IS", "YYLGD.IS", "ZOREN.IS",
]


CONTINUOUS_FEATURES = [
    "ralpha21",
    "mom126_21",
    "mom63_5",
    "roc21",
    "rolling_sharpe21",
    "adx14",
    "obv_slope20",
    "cmf21",
    "rel_volume20",
    "vwap20_slope5",
    "vwap63_slope5",
    "dist_vwap20_atr",
    "dist_vwap63_atr",
    "dist_poc_atr",
    "value_position",
    "profile_width_atr",
    "poc_migration5_atr",
    "turnover_rank_feature",
]

BINARY_FEATURES = [
    "close_above_ema20",
    "ema_stack",
    "close_above_vwap20",
    "close_above_vwap63",
    "close_above_poc",
    "close_above_vah",
    "accepted_above_vah",
    "failed_vah_breakout",
]

FEATURE_COLUMNS = CONTINUOUS_FEATURES + BINARY_FEATURES


@dataclass
class BacktestConfig:
    output_dir: Path
    start_fetch: str = "2018-01-01"
    test_start: str = "2021-05-20"
    end_date: str = "2026-05-21"
    benchmark: str = "XU100.IS"
    holding_label_days: int = 10
    rebalance_every_n_days: int = 30
    top_n: int = 5
    sell_rank_threshold: int = 15
    cost_per_turnover: float = 0.0025
    use_vwap_execution_model: bool = True
    portfolio_value_tl: float = 10_000_000.0
    adv_window: int = 20
    participation_rate: float = 0.03
    max_liquidity_cap_weight: float = 0.35
    dynamic_slippage: bool = True
    base_slippage_bps: float = 3.0
    participation_slippage_bps: float = 18.0
    vol_slippage_bps: float = 12.0
    vwap_chase_slippage_bps: float = 6.0
    profile_width_slippage_bps: float = 2.0
    max_dynamic_slippage_bps: float = 150.0
    use_completion_sleeve: bool = False
    completion_target_weight: float = 1.0
    completion_top_n: int = 8
    completion_candidate_pool: int = 40
    completion_max_position_weight: float = 0.12
    completion_min_cash_to_fill: float = 0.03
    completion_liquidity_weight: float = 0.55
    completion_alpha_weight: float = 0.30
    completion_low_vol_weight: float = 0.15
    completion_max_pairwise_corr: float = 0.90
    factor_beta_window: int = 120
    ridge_lambda: float = 10.0
    profile_window: int = 63
    profile_bins: int = 24
    vol_weight_window: int = 20
    max_position_weight: float = 0.35
    weighting: str = "ml_dynamic"
    ml_weight_power: float = 0.05
    cost_adjust_score: bool = True
    liquidity_cost_penalty: float = 0.00
    vwap_chase_penalty: float = 0.00
    correlation_filter: bool = True
    max_pairwise_corr: float = 0.80
    corr_window: int = 60
    vol_target: float = 0.38
    vol_target_window: int = 40
    min_exposure: float = 0.25
    max_exposure: float = 1.15
    drawdown_governor: bool = True
    dd_level1: float = -0.12
    dd_level2: float = -0.25
    dd_mult1: float = 0.60
    dd_mult2: float = 0.35
    use_meta_label: bool = False
    meta_prob_threshold: float = 0.52
    meta_score_power: float = 1.0
    meta_profit_atr: float = 2.0
    meta_stop_atr: float = 1.0
    regime_filter: bool = False
    benchmark_ema_fast: int = 50
    benchmark_ema_slow: int = 200
    benchmark_vol_window: int = 20
    benchmark_vol_percentile_window: int = 252
    benchmark_vol_percentile_cutoff: float = 0.90
    benchmark_bad_return_window: int = 5
    benchmark_bad_return_threshold: float = -0.08
    benchmark_severe_return_threshold: float = -0.10
    regime_bad_mult: float = 0.80
    regime_severe_mult: float = 0.55
    event_exit: bool = False
    exit_on_below_vwap20: bool = True
    exit_on_below_poc: bool = True
    exit_on_failed_vah: bool = True
    exit_on_nonpositive_score: bool = True
    use_momentum_decay_exit: bool = False
    decay_reduce_threshold: float = 0.45
    decay_exit_threshold: float = 0.70
    decay_reduce_multiplier: float = 0.50
    decay_rank_threshold: int = 0
    decay_rank_weight: float = 0.25
    decay_below_vwap20_weight: float = 0.18
    decay_below_vwap63_weight: float = 0.12
    decay_below_poc_weight: float = 0.15
    decay_ema_break_weight: float = 0.15
    decay_negative_momentum_weight: float = 0.15
    decay_failed_vah_weight: float = 0.20
    effective_trials: int = 288
    max_workers: int = 6
    refresh_data: bool = False
    universe_source: str = "builtin"
    universe_file: Optional[Path] = None
    normal_share_only: bool = True
    data_source: str = "auto"
    tradingview_data_dir: Optional[Path] = None
