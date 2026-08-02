"""Walk-forward ridge model and dynamic feature-weight estimation."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, BacktestConfig


def fit_ridge(X: np.ndarray, y: np.ndarray, ridge_lambda: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    Xs = np.nan_to_num(Xs, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    X_design = np.column_stack([np.ones(len(Xs)), Xs])
    penalty = np.eye(X_design.shape[1]) * ridge_lambda
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(X_design.T @ X_design + penalty, X_design.T @ y)
    return beta, mean, std


def predict_ridge(X: np.ndarray, beta: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Xs = (X - mean) / std
    Xs = np.nan_to_num(Xs, nan=0.0)
    X_design = np.column_stack([np.ones(len(Xs)), Xs])
    return X_design @ beta


def train_walk_forward_scores(panel: pd.DataFrame, cfg: BacktestConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    xcols = [f"x_{c}" for c in FEATURE_COLUMNS]
    panel = panel.sort_values(["date", "symbol"]).copy()
    panel["ml_score"] = np.nan
    panel["meta_prob"] = np.nan
    test_start = pd.Timestamp(cfg.test_start)
    test_end = pd.Timestamp(cfg.end_date)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    prediction_months = pd.period_range(test_start, test_end, freq="M")
    coef_records = []

    for month in prediction_months:
        month_start = month.start_time
        month_end = month.end_time
        if month_end < test_start:
            continue
        train_cutoff_candidates = dates[dates < month_start]
        if len(train_cutoff_candidates) <= cfg.holding_label_days + 260:
            continue
        train_cutoff = train_cutoff_candidates[-(cfg.holding_label_days + 1)]
        train_mask = (
            (panel["date"] <= train_cutoff)
            & panel["label_ml"].notna()
            & panel[xcols].notna().sum(axis=1).ge(len(xcols) - 2)
        )
        train = panel.loc[train_mask]
        if len(train) < 1000:
            continue
        X_train = train[xcols].to_numpy(dtype=float)
        y_train = train["label_ml"].clip(-3.0, 3.0).to_numpy(dtype=float)
        beta, mean, std = fit_ridge(X_train, y_train, cfg.ridge_lambda)

        pred_mask = (panel["date"] >= max(month_start, test_start)) & (panel["date"] <= min(month_end, test_end))
        pred_idx = panel.index[pred_mask]
        if len(pred_idx) == 0:
            continue
        X_pred = panel.loc[pred_idx, xcols].to_numpy(dtype=float)
        panel.loc[pred_idx, "ml_score"] = predict_ridge(X_pred, beta, mean, std)

        meta_train_mask = train_mask & panel["label_meta"].notna()
        meta_train = panel.loc[meta_train_mask]
        meta_train_rows = len(meta_train)
        meta_beta = None
        if cfg.use_meta_label and meta_train_rows >= 1000 and meta_train["label_meta"].nunique() > 1:
            X_meta = meta_train[xcols].to_numpy(dtype=float)
            y_meta = meta_train["label_meta"].clip(0.0, 1.0).to_numpy(dtype=float)
            meta_beta, meta_mean, meta_std = fit_ridge(X_meta, y_meta, cfg.ridge_lambda)
            meta_raw = predict_ridge(X_pred, meta_beta, meta_mean, meta_std)
            panel.loc[pred_idx, "meta_prob"] = np.clip(meta_raw, 0.01, 0.99)
        else:
            panel.loc[pred_idx, "meta_prob"] = 1.0

        record = {
            "month": str(month),
            "intercept": beta[0],
            "train_rows": len(train),
            "meta_train_rows": meta_train_rows,
            "train_cutoff": str(train_cutoff.date()),
        }
        for col, coef in zip(FEATURE_COLUMNS, beta[1:]):
            record[col] = coef
        if meta_beta is not None:
            record["meta_intercept"] = meta_beta[0]
            for col, coef in zip(FEATURE_COLUMNS, meta_beta[1:]):
                record[f"meta_{col}"] = coef
        coef_records.append(record)
        print(f"[ML] {month}: train_rows={len(train)} meta_rows={meta_train_rows} cutoff={train_cutoff.date()}")

    coef_df = pd.DataFrame(coef_records)
    return panel, coef_df
