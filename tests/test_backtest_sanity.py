#!/usr/bin/env python3
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vwap_profile_ml_backtest as bt


def make_minimal_panel() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": ["AAA.IS"] * 3,
            "open": [100.0, 200.0, 220.0],
            "close": [100.0, 200.0, 220.0],
            "ml_score": [1.0, 1.0, 1.0],
            "meta_prob": [1.0, 1.0, 1.0],
            "failed_vah_breakout": [0.0, 0.0, 0.0],
            "dist_vwap20_atr": [0.0, 0.0, 0.0],
            "close_above_vwap20": [1.0, 1.0, 1.0],
            "close_above_poc": [1.0, 1.0, 1.0],
            "profile_width_atr": [1.0, 1.0, 1.0],
            "x_turnover_rank_feature": [0.0, 0.0, 0.0],
        }
    )


def make_benchmark() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "adj_close": [100.0, 100.0, 100.0],
            "volume": [1.0, 1.0, 1.0],
            "symbol": ["XU100.IS"] * 3,
        }
    )


def make_two_asset_panel() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    rows = []
    prices = {
        "AAA.IS": {
            "open": [100.0, 100.0, 220.0],
            "close": [100.0, 200.0, 220.0],
        },
        "BBB.IS": {
            "open": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
        },
    }
    for symbol, px in prices.items():
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": px["open"][idx],
                    "close": px["close"][idx],
                    "ml_score": 1.0,
                    "meta_prob": 1.0,
                    "failed_vah_breakout": 0.0,
                    "dist_vwap20_atr": 0.0,
                    "close_above_vwap20": 1.0,
                    "close_above_poc": 1.0,
                    "profile_width_atr": 1.0,
                    "x_turnover_rank_feature": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_next_open_execution_excludes_overnight_gap() -> None:
    panel = make_minimal_panel()
    cfg = bt.BacktestConfig(
        output_dir=Path("tmp"),
        test_start="2024-01-02",
        end_date="2024-01-04",
        top_n=1,
        rebalance_every_n_days=1,
        sell_rank_threshold=1,
        cost_per_turnover=0.0,
        weighting="equal",
        correlation_filter=False,
        vol_target=0.0,
        drawdown_governor=False,
        use_vwap_execution_model=False,
    )
    equity, _, _ = bt.run_backtest(panel, make_benchmark(), cfg)
    # Signal is known after 2024-01-02 close. The 2024-01-03 open gaps from
    # 100 to 200, but a next-open execution cannot capture that gap.
    assert abs(equity.loc[pd.Timestamp("2024-01-03"), "strategy_return"]) < 1e-12


def test_next_open_execution_captures_intraday_after_entry() -> None:
    panel = make_minimal_panel()
    panel.loc[panel["date"].eq(pd.Timestamp("2024-01-03")), "close"] = 220.0
    cfg = bt.BacktestConfig(
        output_dir=Path("tmp"),
        test_start="2024-01-02",
        end_date="2024-01-04",
        top_n=1,
        rebalance_every_n_days=1,
        sell_rank_threshold=1,
        cost_per_turnover=0.0,
        weighting="equal",
        correlation_filter=False,
        vol_target=0.0,
        drawdown_governor=False,
        use_vwap_execution_model=False,
    )
    equity, _, _ = bt.run_backtest(panel, make_benchmark(), cfg)
    assert abs(equity.loc[pd.Timestamp("2024-01-03"), "strategy_return"] - 0.10) < 1e-12


def test_close_weight_drift_feeds_next_overnight_return() -> None:
    cfg = bt.BacktestConfig(
        output_dir=Path("tmp"),
        test_start="2024-01-02",
        end_date="2024-01-04",
        top_n=2,
        rebalance_every_n_days=10,
        sell_rank_threshold=2,
        cost_per_turnover=0.0,
        weighting="equal",
        correlation_filter=False,
        vol_target=0.0,
        drawdown_governor=False,
        use_vwap_execution_model=False,
    )
    equity, _, _ = bt.run_backtest(make_two_asset_panel(), make_benchmark(), cfg)
    # Day 2 close weights drift to 2/3 AAA and 1/3 BBB after AAA doubles intraday.
    # The next overnight AAA gap is therefore captured at a 2/3 weight, not 1/2.
    expected = (2.0 / 3.0) * 0.10
    assert abs(equity.loc[pd.Timestamp("2024-01-04"), "strategy_return"] - expected) < 1e-12


def test_no_daily_rebalance_when_signal_is_unchanged() -> None:
    cfg = bt.BacktestConfig(
        output_dir=Path("tmp"),
        test_start="2024-01-02",
        end_date="2024-01-04",
        top_n=2,
        rebalance_every_n_days=10,
        sell_rank_threshold=2,
        cost_per_turnover=0.0,
        weighting="equal",
        correlation_filter=False,
        vol_target=0.0,
        drawdown_governor=False,
        use_vwap_execution_model=False,
    )
    equity, _, _ = bt.run_backtest(make_two_asset_panel(), make_benchmark(), cfg)
    assert abs(equity.loc[pd.Timestamp("2024-01-04"), "turnover"]) < 1e-12


def test_event_exit_executes_next_open_not_same_close() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    panel = make_minimal_panel()
    panel["open"] = [100.0, 100.0, 50.0]
    panel["close"] = [100.0, 100.0, 50.0]
    panel["close_above_vwap20"] = [1.0, 0.0, 0.0]
    cfg = bt.BacktestConfig(
        output_dir=Path("tmp"),
        test_start="2024-01-02",
        end_date="2024-01-04",
        top_n=1,
        rebalance_every_n_days=10,
        sell_rank_threshold=1,
        cost_per_turnover=0.0,
        weighting="equal",
        correlation_filter=False,
        vol_target=0.0,
        drawdown_governor=False,
        use_vwap_execution_model=False,
        regime_filter=False,
        use_meta_label=False,
        event_exit=True,
        exit_on_below_vwap20=True,
        exit_on_below_poc=False,
        exit_on_failed_vah=False,
        exit_on_nonpositive_score=False,
    )
    equity, _, _ = bt.run_backtest(panel, make_benchmark(), cfg)
    assert abs(equity.loc[dates[1], "strategy_return"]) < 1e-12
    assert abs(equity.loc[dates[2], "strategy_return"] + 0.50) < 1e-12
    assert equity.loc[dates[2], "active_positions"] == 0


def test_momentum_decay_exit_reduces_position() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    panel = make_minimal_panel()
    panel["close_above_vwap20"] = [1.0, 0.0, 0.0]
    panel["close_above_vwap63"] = [1.0, 1.0, 1.0]
    panel["close_above_poc"] = [1.0, 0.0, 0.0]
    panel["close_above_ema20"] = [1.0, 0.0, 0.0]
    panel["ema_stack"] = [1.0, 1.0, 1.0]
    panel["mom63_5"] = [0.1, 0.1, 0.1]
    panel["mom126_21"] = [0.1, 0.1, 0.1]
    cfg = bt.BacktestConfig(
        output_dir=Path("tmp"),
        test_start="2024-01-02",
        end_date="2024-01-04",
        top_n=1,
        rebalance_every_n_days=10,
        sell_rank_threshold=1,
        cost_per_turnover=0.0,
        weighting="equal",
        correlation_filter=False,
        vol_target=0.0,
        drawdown_governor=False,
        use_vwap_execution_model=False,
        use_momentum_decay_exit=True,
        decay_reduce_threshold=0.40,
        decay_exit_threshold=0.80,
        decay_reduce_multiplier=0.50,
    )
    _, _, summary = bt.run_backtest(panel, make_benchmark(), cfg)
    actions = summary.get("exit_action_counts", {})
    assert actions.get("reduce", 0) >= 1
    records = summary["_event_exit_records"]
    assert any(record["action"] == "reduce" and abs(record["new_weight"] - 0.5) < 1e-12 for record in records)


def test_weight_cap_is_respected_after_renormalization() -> None:
    weights = pd.Series({"A": 0.90, "B": 0.08, "C": 0.02})
    capped = bt.cap_and_renormalize_weights(weights, 0.50)
    assert abs(capped.sum() - 1.0) < 1e-12
    assert capped.max() <= 0.50 + 1e-12


def test_vwap_liquidity_cap_reduces_size_without_min_volume_filter() -> None:
    weights = pd.Series({"AAA.IS": 0.50, "BBB.IS": 0.50})
    adv_tl = pd.Series({"AAA.IS": 100_000_000.0, "BBB.IS": 100_000.0})
    cfg = bt.BacktestConfig(
        output_dir=Path("tmp"),
        use_vwap_execution_model=True,
        portfolio_value_tl=10_000_000.0,
        participation_rate=0.03,
        max_liquidity_cap_weight=0.35,
    )
    capped = bt.apply_vwap_liquidity_caps(weights, adv_tl, cfg)
    assert 0.0 < capped.loc["BBB.IS"] < weights.loc["BBB.IS"]
    assert capped.sum() < weights.sum()


def test_completion_sleeve_fills_cash_with_liquid_names_and_respects_caps() -> None:
    alpha_weights = pd.Series({"AAA.IS": 0.50})
    ranked = pd.Series({"AAA.IS": 1.00, "BBB.IS": 0.90, "CCC.IS": 0.80, "DDD.IS": 0.70})
    trailing_vol = pd.Series({"BBB.IS": 0.02, "CCC.IS": 0.03, "DDD.IS": 0.01})
    adv_tl = pd.Series({"AAA.IS": 1_000_000_000.0, "BBB.IS": 200_000_000.0, "CCC.IS": 180_000_000.0, "DDD.IS": 1_000_000.0})
    cfg = bt.BacktestConfig(
        output_dir=Path("tmp"),
        use_completion_sleeve=True,
        use_vwap_execution_model=True,
        portfolio_value_tl=10_000_000.0,
        participation_rate=0.03,
        completion_top_n=2,
        completion_candidate_pool=3,
        completion_max_position_weight=0.30,
    )
    completion = bt.build_completion_sleeve(
        alpha_weights=alpha_weights,
        ranked=ranked,
        trailing_vol_row=trailing_vol,
        adv_tl_row=adv_tl,
        corr_matrix=None,
        cfg=cfg,
    )
    assert abs(completion.sum() - 0.50) < 1e-12
    assert set(completion.index) == {"BBB.IS", "CCC.IS"}
    assert completion.max() <= 0.30 + 1e-12


def test_dynamic_slippage_increases_with_trade_participation() -> None:
    delta = pd.Series({"AAA.IS": 0.10})
    cfg = bt.BacktestConfig(
        output_dir=Path("tmp"),
        cost_per_turnover=0.0,
        use_vwap_execution_model=True,
        dynamic_slippage=True,
        portfolio_value_tl=10_000_000.0,
    )
    common = {
        "delta_weights": delta,
        "atr_pct_row": pd.Series({"AAA.IS": 0.03}),
        "dist_vwap_row": pd.Series({"AAA.IS": 0.0}),
        "profile_width_row": pd.Series({"AAA.IS": 1.0}),
        "cfg": cfg,
        "equity_after_overnight": 1.0,
    }
    high_adv = bt.estimate_vwap_execution_cost_rate(adv_tl_row=pd.Series({"AAA.IS": 100_000_000.0}), **common)
    low_adv = bt.estimate_vwap_execution_cost_rate(adv_tl_row=pd.Series({"AAA.IS": 1_000_000.0}), **common)
    assert low_adv["dynamic_slippage_rate"] > high_adv["dynamic_slippage_rate"]
    assert low_adv["max_trade_participation"] > high_adv["max_trade_participation"]


def test_volume_profile_uses_only_prior_window() -> None:
    typical = np.array([10.0, 10.0, 10.0, 100.0])
    high = typical + 0.1
    low = typical - 0.1
    volume = np.array([1.0, 1.0, 1.0, 1000.0])
    poc, _, _ = bt.compute_volume_profile_levels(typical, high, low, volume, window=3, n_bins=10)
    # At index 3 the current 100-price, high-volume outlier must not affect POC.
    assert poc[3] < 11.0


def test_triple_barrier_uses_future_path_after_signal_day() -> None:
    idx = pd.date_range("2024-01-01", periods=5)
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=idx)
    high = pd.Series([100.0, 100.5, 102.5, 100.0, 100.0], index=idx)
    low = pd.Series([99.8, 99.8, 99.8, 99.8, 99.8], index=idx)
    atr = pd.Series([1.0] * 5, index=idx)
    label = bt.triple_barrier_meta_label(close, high, low, atr, horizon=2, profit_atr=2.0, stop_atr=1.0)
    assert label.iloc[0] == 1.0


def test_universe_loader_filters_non_normal_instruments() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "universe.csv"
        pd.DataFrame(
            [
                {"symbol": "AKBNK", "instrument_type": "Pay", "market": "Yildiz Pazar"},
                {"symbol": "ETFTR", "instrument_type": "Borsa Yatirim Fonu", "market": "Fon Pazari"},
                {"symbol": "VRNT", "instrument_type": "Varant", "market": "Varant Pazari"},
                {"symbol": "RUCHN", "instrument_type": "Ruchan Kuponu", "market": "Pay Piyasasi"},
            ]
        ).to_csv(path, index=False)
        cfg = bt.BacktestConfig(output_dir=Path(tmp), universe_file=path, normal_share_only=True)
        assert bt.load_universe_symbols(cfg) == ["AKBNK.IS"]


def test_tradingview_export_loader_parses_common_columns() -> None:
    with TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        pd.DataFrame(
            {
                "time": ["2024-01-02", "2024-01-03"],
                "open": [10.0, 11.0],
                "high": [11.0, 12.0],
                "low": [9.5, 10.5],
                "close": [10.5, 11.5],
                "Volume": [1000, 2000],
            }
        ).to_csv(data_dir / "BIST_AAAA.csv", index=False)
        cfg = bt.BacktestConfig(
            output_dir=Path(tmp),
            start_fetch="2024-01-01",
            end_date="2024-01-10",
            tradingview_data_dir=data_dir,
            data_source="tradingview",
        )
        df = bt.load_tradingview_export("AAAA.IS", cfg)
        assert df is not None
        assert list(df["close"]) == [10.5, 11.5]
        assert df["data_source"].iloc[0] == "tradingview_export"


def test_tradingview_universe_filters_altin_certificate() -> None:
    row = pd.Series(
        {
            "symbol": "ALTIN.IS",
            "name": "DARPHANE ALTIN SERTIFIKASI",
            "description": "DARPHANE ALTIN SERTIFIKASI",
            "instrument_type": "stock",
            "subtype": "common",
            "market": "turkey",
        }
    )
    assert not bt.is_normal_share_row(row, True)


if __name__ == "__main__":
    tests = [
        test_next_open_execution_excludes_overnight_gap,
        test_next_open_execution_captures_intraday_after_entry,
        test_close_weight_drift_feeds_next_overnight_return,
        test_no_daily_rebalance_when_signal_is_unchanged,
        test_event_exit_executes_next_open_not_same_close,
        test_momentum_decay_exit_reduces_position,
        test_weight_cap_is_respected_after_renormalization,
        test_vwap_liquidity_cap_reduces_size_without_min_volume_filter,
        test_completion_sleeve_fills_cash_with_liquid_names_and_respects_caps,
        test_dynamic_slippage_increases_with_trade_participation,
        test_volume_profile_uses_only_prior_window,
        test_triple_barrier_uses_future_path_after_signal_day,
        test_universe_loader_filters_non_normal_instruments,
        test_tradingview_export_loader_parses_common_columns,
        test_tradingview_universe_filters_altin_certificate,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
