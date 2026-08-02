"""Portfolio construction, liquidity capacity, and execution-cost logic."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import BacktestConfig


def select_low_correlation_names(
    ranked: pd.Series,
    keep: List[str],
    top_n: int,
    corr_matrix: Optional[pd.DataFrame],
    max_pairwise_corr: float,
) -> List[str]:
    selected: List[str] = []

    def corr_ok(symbol: str, existing: List[str]) -> bool:
        if not existing or corr_matrix is None or symbol not in corr_matrix.index:
            return True
        existing = [s for s in existing if s in corr_matrix.columns]
        if not existing:
            return True
        corr = corr_matrix.loc[symbol, existing].abs().replace([np.inf, -np.inf], np.nan).dropna()
        if corr.empty:
            return True
        return bool(corr.max() <= max_pairwise_corr)

    for symbol in keep:
        if symbol in ranked.index and corr_ok(symbol, selected):
            selected.append(symbol)
        if len(selected) >= top_n:
            return selected

    for symbol in ranked.index:
        if symbol in selected:
            continue
        if corr_ok(symbol, selected):
            selected.append(symbol)
        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        for symbol in ranked.index:
            if symbol not in selected:
                selected.append(symbol)
            if len(selected) >= top_n:
                break

    return selected


def cap_and_renormalize_weights(weights: pd.Series, max_weight: float) -> pd.Series:
    weights = weights.dropna().clip(lower=0.0)
    if weights.empty or weights.sum() <= 0:
        return weights
    weights = weights / weights.sum()
    if max_weight <= 0 or max_weight >= 1 or len(weights) == 1:
        return weights
    if max_weight * len(weights) < 1.0:
        max_weight = 1.0 / len(weights)

    capped = pd.Series(0.0, index=weights.index)
    free = weights.copy()
    remaining_mass = 1.0

    while not free.empty:
        scaled = free / free.sum() * remaining_mass
        over = scaled > max_weight
        if not over.any():
            capped.loc[scaled.index] = scaled
            break
        capped_names = scaled[over].index
        capped.loc[capped_names] = max_weight
        remaining_mass -= max_weight * len(capped_names)
        free = free.drop(capped_names)
        if remaining_mass <= 1e-12:
            break

    if capped.sum() > 0:
        capped = capped / capped.sum()
    return capped


def optional_pivot(panel: pd.DataFrame, template: pd.DataFrame, column: str, default: float = np.nan) -> pd.DataFrame:
    if column not in panel.columns:
        return pd.DataFrame(default, index=template.index, columns=template.columns)
    return panel.pivot(index="date", columns="symbol", values=column).sort_index().reindex_like(template)


def cap_weights_with_symbol_caps(weights: pd.Series, caps: pd.Series) -> pd.Series:
    weights = weights.dropna().clip(lower=0.0)
    if weights.empty or weights.sum() <= 0:
        return weights

    target_total = float(weights.sum())
    caps = pd.to_numeric(caps.reindex(weights.index), errors="coerce")
    caps = caps.where(caps.notna(), np.inf).clip(lower=0.0)
    finite_caps = caps.replace(np.inf, target_total)
    feasible_total = min(target_total, float(finite_caps.sum()))
    if feasible_total <= 0:
        return pd.Series(0.0, index=weights.index)
    if (weights <= caps).all() and abs(feasible_total - target_total) < 1e-12:
        return weights

    capped = pd.Series(0.0, index=weights.index)
    free = weights.copy()
    remaining_mass = feasible_total

    while not free.empty and remaining_mass > 1e-12:
        scaled = free / free.sum() * remaining_mass
        free_caps = caps.reindex(scaled.index)
        over = scaled > free_caps
        if not over.any():
            capped.loc[scaled.index] = scaled
            break
        capped_names = scaled[over].index
        capped.loc[capped_names] = free_caps.loc[capped_names]
        remaining_mass -= float(free_caps.loc[capped_names].sum())
        free = free.drop(capped_names)

    return capped


def compute_liquidity_weight_caps(adv_tl_row: pd.Series, cfg: BacktestConfig) -> pd.Series:
    caps = pd.Series(np.inf, index=adv_tl_row.index, dtype=float)
    if (
        not cfg.use_vwap_execution_model
        or cfg.portfolio_value_tl <= 0
        or cfg.participation_rate <= 0
    ):
        return caps

    adv = pd.to_numeric(adv_tl_row, errors="coerce")
    valid_adv = adv.notna()
    caps.loc[valid_adv] = adv.loc[valid_adv].clip(lower=0.0) * cfg.participation_rate / cfg.portfolio_value_tl
    if cfg.max_liquidity_cap_weight > 0:
        caps.loc[valid_adv] = caps.loc[valid_adv].clip(upper=cfg.max_liquidity_cap_weight)
    return caps


def apply_vwap_liquidity_caps(weights: pd.Series, adv_tl_row: pd.Series, cfg: BacktestConfig) -> pd.Series:
    if not cfg.use_vwap_execution_model or weights.empty:
        return weights
    caps = compute_liquidity_weight_caps(adv_tl_row.reindex(weights.index), cfg)
    return cap_weights_with_symbol_caps(weights, caps)


def weighted_factor_exposure(weights: pd.Series, factor_row: pd.Series) -> float:
    active = weights.replace([np.inf, -np.inf], np.nan).dropna()
    active = active[active.abs() > 0]
    if active.empty:
        return np.nan
    factors = pd.to_numeric(factor_row.reindex(active.index), errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = factors.notna()
    if not mask.any():
        return np.nan
    active = active.loc[mask]
    factors = factors.loc[mask]
    denom = float(active.abs().sum())
    if denom <= 0:
        return np.nan
    return float((active * factors).sum() / denom)


def build_completion_sleeve(
    alpha_weights: pd.Series,
    ranked: pd.Series,
    trailing_vol_row: pd.Series,
    adv_tl_row: pd.Series,
    corr_matrix: Optional[pd.DataFrame],
    cfg: BacktestConfig,
) -> pd.Series:
    alpha_weights = alpha_weights.dropna().clip(lower=0.0)
    if not cfg.use_completion_sleeve:
        return pd.Series(dtype=float)

    target_weight = float(np.clip(cfg.completion_target_weight, 0.0, 1.0))
    residual = target_weight - float(alpha_weights.sum())
    if residual <= max(cfg.completion_min_cash_to_fill, 1e-12):
        return pd.Series(dtype=float)

    candidates = ranked.drop(index=alpha_weights[alpha_weights > 0].index, errors="ignore")
    candidates = candidates.replace([np.inf, -np.inf], np.nan).dropna()
    candidates = candidates[candidates > 0.0]
    if candidates.empty:
        return pd.Series(dtype=float)

    adv = pd.to_numeric(adv_tl_row.reindex(candidates.index), errors="coerce").replace([np.inf, -np.inf], np.nan)
    candidates = candidates.loc[adv.notna() & adv.gt(0.0)]
    if candidates.empty:
        return pd.Series(dtype=float)
    adv = adv.reindex(candidates.index)

    pool_size = max(cfg.completion_top_n, cfg.completion_candidate_pool)
    pool_symbols = adv.sort_values(ascending=False).head(pool_size).index
    candidates = candidates.loc[pool_symbols]
    adv = adv.loc[pool_symbols]

    score_rank = candidates.rank(pct=True)
    liquidity_rank = adv.rank(pct=True)
    vol = pd.to_numeric(trailing_vol_row.reindex(candidates.index), errors="coerce").replace([np.inf, -np.inf], np.nan)
    low_vol_rank = (-vol.fillna(vol.median()).fillna(0.0)).rank(pct=True)
    composite = (
        cfg.completion_liquidity_weight * liquidity_rank
        + cfg.completion_alpha_weight * score_rank
        + cfg.completion_low_vol_weight * low_vol_rank
    ).replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    if composite.empty:
        return pd.Series(dtype=float)

    selected_symbols = select_low_correlation_names(
        ranked=composite,
        keep=[],
        top_n=max(1, cfg.completion_top_n),
        corr_matrix=corr_matrix,
        max_pairwise_corr=cfg.completion_max_pairwise_corr,
    )
    selected = composite.loc[selected_symbols].sort_values(ascending=False)
    selected_vol = vol.reindex(selected.index).replace(0.0, np.nan)
    selected_vol = selected_vol.fillna(selected_vol.median()).fillna(1.0)
    raw = selected.clip(lower=0.0) / selected_vol
    if raw.sum() <= 0:
        raw = pd.Series(1.0, index=selected.index)
    raw_weights = raw / raw.sum() * residual

    position_caps = pd.Series(cfg.completion_max_position_weight, index=raw_weights.index, dtype=float)
    if cfg.use_vwap_execution_model:
        liquidity_caps = compute_liquidity_weight_caps(adv_tl_row.reindex(raw_weights.index), cfg)
        remaining_caps = liquidity_caps - alpha_weights.reindex(raw_weights.index).fillna(0.0)
        position_caps = pd.concat([position_caps, remaining_caps.clip(lower=0.0)], axis=1).min(axis=1)
    return cap_weights_with_symbol_caps(raw_weights, position_caps)


def estimate_vwap_execution_cost_rate(
    delta_weights: pd.Series,
    adv_tl_row: pd.Series,
    atr_pct_row: pd.Series,
    dist_vwap_row: pd.Series,
    profile_width_row: pd.Series,
    cfg: BacktestConfig,
    equity_after_overnight: float,
) -> Dict[str, float]:
    abs_delta = delta_weights.abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    turnover_value = float(abs_delta.sum())
    base_cost_rate = turnover_value * cfg.cost_per_turnover
    result = {
        "total_cost_rate": base_cost_rate,
        "dynamic_slippage_rate": 0.0,
        "avg_trade_participation": 0.0,
        "max_trade_participation": 0.0,
    }
    if (
        not cfg.use_vwap_execution_model
        or not cfg.dynamic_slippage
        or turnover_value <= 0
        or cfg.portfolio_value_tl <= 0
    ):
        return result

    symbols = abs_delta.index
    portfolio_notional_tl = max(float(cfg.portfolio_value_tl), 0.0)
    trade_tl = abs_delta * portfolio_notional_tl
    adv = pd.to_numeric(adv_tl_row.reindex(symbols), errors="coerce")
    valid_adv = adv.where(adv > 0.0)
    participation = (trade_tl / valid_adv).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

    participation_base = max(cfg.participation_rate, 1e-12)
    participation_bps = cfg.participation_slippage_bps * np.sqrt((participation / participation_base).clip(0.0, 25.0))

    atr_pct = pd.to_numeric(atr_pct_row.reindex(symbols), errors="coerce").replace([np.inf, -np.inf], np.nan)
    vol_bps = cfg.vol_slippage_bps * (atr_pct.fillna(0.0).clip(0.0, 0.20) / 0.03)

    dist = pd.to_numeric(dist_vwap_row.reindex(symbols), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    directional_chase = pd.Series(
        np.where(delta_weights.reindex(symbols).fillna(0.0) >= 0.0, dist.clip(lower=0.0), (-dist).clip(lower=0.0)),
        index=symbols,
    )
    vwap_bps = cfg.vwap_chase_slippage_bps * directional_chase.clip(0.0, 3.0)

    profile_width = pd.to_numeric(profile_width_row.reindex(symbols), errors="coerce")
    profile_bps = cfg.profile_width_slippage_bps * profile_width.fillna(0.0).clip(0.0, 5.0)

    slippage_bps = cfg.base_slippage_bps + participation_bps + vol_bps + vwap_bps + profile_bps
    if cfg.max_dynamic_slippage_bps > 0:
        slippage_bps = slippage_bps.clip(upper=cfg.max_dynamic_slippage_bps)
    dynamic_slippage_rate = float((abs_delta * slippage_bps / 10_000.0).sum())

    if turnover_value > 0:
        avg_participation = float((abs_delta * participation).sum() / turnover_value)
    else:
        avg_participation = 0.0
    result.update(
        {
            "total_cost_rate": base_cost_rate + dynamic_slippage_rate,
            "dynamic_slippage_rate": dynamic_slippage_rate,
            "avg_trade_participation": avg_participation,
            "max_trade_participation": float(participation.max()) if len(participation) else 0.0,
        }
    )
    return result


def compute_portfolio_weights(
    selected: pd.Series,
    trailing_vol_row: pd.Series,
    corr_matrix: Optional[pd.DataFrame],
    cfg: BacktestConfig,
) -> pd.Series:
    if len(selected) == 0:
        return pd.Series(dtype=float)

    if cfg.weighting == "equal":
        weights = pd.Series(1.0 / len(selected), index=selected.index)
    elif cfg.weighting == "score":
        raw_weights = selected.clip(lower=0.0)
        if raw_weights.sum() <= 0:
            raw_weights = pd.Series(1.0, index=selected.index)
        weights = raw_weights / raw_weights.sum()
    elif cfg.weighting == "invvol":
        vol = trailing_vol_row.reindex(selected.index).replace(0, np.nan)
        vol = vol.fillna(vol.median()).fillna(1.0)
        raw_weights = selected.clip(lower=0.0) / vol
        if raw_weights.sum() <= 0:
            raw_weights = pd.Series(1.0, index=selected.index)
        weights = raw_weights / raw_weights.sum()
    elif cfg.weighting == "ml_dynamic":
        raw_weights = selected.clip(lower=0.0).pow(cfg.ml_weight_power)
        if corr_matrix is not None and len(selected) > 1:
            penalties = {}
            for symbol in selected.index:
                peers = [s for s in selected.index if s != symbol]
                if symbol in corr_matrix.index and peers:
                    avg_corr = corr_matrix.loc[symbol, peers].abs().replace([np.inf, -np.inf], np.nan).mean()
                    penalties[symbol] = 1.0 + (0.0 if np.isnan(avg_corr) else float(avg_corr))
                else:
                    penalties[symbol] = 1.0
            raw_weights = raw_weights / pd.Series(penalties)
        if raw_weights.sum() <= 0:
            raw_weights = pd.Series(1.0, index=selected.index)
        weights = raw_weights / raw_weights.sum()
    else:
        raise ValueError(f"Unknown weighting mode: {cfg.weighting}")

    return cap_and_renormalize_weights(weights, cfg.max_position_weight)
