"""Daily OHLCV ingestion from Yahoo Finance or TradingView exports."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests

from .config import BacktestConfig
from .universe import normalize_text, to_epoch


def fetch_yahoo_chart(symbol: str, start: str, end: str, timeout: int = 20) -> Optional[pd.DataFrame]:
    period1 = to_epoch(start)
    period2 = to_epoch(end)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}"
                time.sleep(0.5 * (attempt + 1))
                continue
            payload = response.json()
            error = payload.get("chart", {}).get("error")
            if error:
                last_error = str(error)
                time.sleep(0.5 * (attempt + 1))
                continue
            result = payload.get("chart", {}).get("result") or []
            if not result:
                last_error = "empty result"
                continue
            result0 = result[0]
            timestamps = result0.get("timestamp") or []
            quote = (result0.get("indicators", {}).get("quote") or [{}])[0]
            adj = (result0.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
            if not timestamps or not quote:
                last_error = "missing timestamps or quotes"
                continue
            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(timestamps, unit="s").tz_localize("UTC").tz_convert("Europe/Istanbul").date,
                    "open": quote.get("open"),
                    "high": quote.get("high"),
                    "low": quote.get("low"),
                    "close": quote.get("close"),
                    "volume": quote.get("volume"),
                }
            )
            df["date"] = pd.to_datetime(df["date"])
            if adj is not None:
                df["adj_close"] = adj
            else:
                df["adj_close"] = df["close"]
            df["symbol"] = symbol
            df = df.dropna(subset=["open", "high", "low", "close"]).drop_duplicates("date")
            if df.empty:
                last_error = "empty dataframe after cleaning"
                continue
            for col in ["open", "high", "low", "close", "adj_close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["volume"] = df["volume"].fillna(0.0)
            df["data_source"] = "yahoo"
            return df.sort_values("date")
        except Exception as exc:  # noqa: BLE001 - intentionally robust network wrapper
            last_error = repr(exc)
            time.sleep(0.5 * (attempt + 1))
    print(f"[WARN] {symbol} fetch failed: {last_error}")
    return None


def tradingview_symbol_base(symbol: str) -> str:
    return symbol.replace(".IS", "").replace(".", "_").upper()


def find_tradingview_export(symbol: str, data_dir: Path) -> Optional[Path]:
    base = tradingview_symbol_base(symbol)
    exact_names = [
        f"{symbol}.csv",
        f"{symbol.replace('.', '_')}.csv",
        f"{base}.csv",
        f"BIST_{base}.csv",
        f"BIST-{base}.csv",
        f"BIST:{base}.csv",
    ]
    for name in exact_names:
        path = data_dir / name
        if path.exists():
            return path

    matches = sorted(data_dir.glob(f"*{base}*.csv"))
    return matches[0] if matches else None


def normalize_ohlcv_columns(columns: Iterable[str]) -> Dict[str, str]:
    normalized = {}
    for col in columns:
        key = normalize_text(col).replace(" ", "").replace("_", "")
        normalized[key] = col
    result = {}
    for target, candidates in {
        "date": ["DATE", "TIME", "DATETIME", "TIMESTAMP"],
        "open": ["OPEN", "O"],
        "high": ["HIGH", "H"],
        "low": ["LOW", "L"],
        "close": ["CLOSE", "C"],
        "volume": ["VOLUME", "VOL"],
    }.items():
        for candidate in candidates:
            key = candidate.replace(" ", "").replace("_", "")
            if key in normalized:
                result[target] = normalized[key]
                break
    return result


def parse_tradingview_dates(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() > 0.90:
        median = float(numeric.dropna().median())
        if median > 1e11:
            return pd.to_datetime(numeric, unit="ms", errors="coerce")
        if median > 1e9:
            return pd.to_datetime(numeric, unit="s", errors="coerce")
    return pd.to_datetime(values, errors="coerce")


def load_tradingview_export(symbol: str, cfg: BacktestConfig) -> Optional[pd.DataFrame]:
    if cfg.tradingview_data_dir is None:
        return None
    data_dir = Path(cfg.tradingview_data_dir)
    path = find_tradingview_export(symbol, data_dir)
    if path is None:
        print(f"[WARN] {symbol} TradingView export not found in {data_dir}")
        return None

    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] {symbol} TradingView export read failed: {exc}")
        return None

    cols = normalize_ohlcv_columns(raw.columns)
    required = {"date", "open", "high", "low", "close"}
    missing = required - set(cols)
    if missing:
        print(f"[WARN] {symbol} TradingView export missing columns {sorted(missing)}: {path}")
        return None

    df = pd.DataFrame(
        {
            "date": parse_tradingview_dates(raw[cols["date"]]),
            "open": pd.to_numeric(raw[cols["open"]], errors="coerce"),
            "high": pd.to_numeric(raw[cols["high"]], errors="coerce"),
            "low": pd.to_numeric(raw[cols["low"]], errors="coerce"),
            "close": pd.to_numeric(raw[cols["close"]], errors="coerce"),
            "volume": pd.to_numeric(raw[cols["volume"]], errors="coerce") if "volume" in cols else 0.0,
        }
    )
    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
    df["date"] = pd.to_datetime(df["date"].dt.date)
    df["volume"] = df["volume"].fillna(0.0)
    df["adj_close"] = df["close"]
    df["symbol"] = symbol
    df["data_source"] = "tradingview_export"
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).drop_duplicates("date").sort_values("date")
    start = pd.Timestamp(cfg.start_fetch)
    end = pd.Timestamp(cfg.end_date)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    return df if not df.empty else None


def load_or_fetch_symbol(symbol: str, cfg: BacktestConfig, raw_dir: Path) -> Optional[pd.DataFrame]:
    path = raw_dir / f"{symbol.replace('.', '_')}.csv"
    if path.exists() and not cfg.refresh_data:
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            cached_source = str(df["data_source"].iloc[0]) if "data_source" in df.columns and not df.empty else "unknown"
            cache_ok = (
                cfg.data_source == "auto"
                or cached_source == cfg.data_source
                or (cfg.data_source == "tradingview" and cached_source == "tradingview_export")
            )
            if not df.empty and cache_ok:
                return df
        except Exception:
            pass

    df = None
    if cfg.data_source in {"auto", "yahoo"}:
        df = fetch_yahoo_chart(symbol, cfg.start_fetch, cfg.end_date)
    if df is None and cfg.data_source in {"auto", "tradingview"}:
        df = load_tradingview_export(symbol, cfg)
    if df is not None and not df.empty:
        df.to_csv(path, index=False)
    return df


def fetch_all_data(symbols: List[str], cfg: BacktestConfig) -> Dict[str, pd.DataFrame]:
    raw_dir = cfg.output_dir / "data_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    data: Dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {executor.submit(load_or_fetch_symbol, s, cfg, raw_dir): s for s in symbols}
        for fut in as_completed(futures):
            symbol = futures[fut]
            df = fut.result()
            if df is not None and len(df) > 260:
                data[symbol] = df
                source = df["data_source"].iloc[0] if "data_source" in df.columns and not df.empty else "unknown"
                print(f"[DATA] {symbol}: {len(df)} rows source={source}")
            else:
                print(f"[SKIP] {symbol}: insufficient data")
    return data


def adjust_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    factor = (out["adj_close"] / out["close"]).replace([np.inf, -np.inf], np.nan)
    factor = factor.fillna(1.0)
    for col in ["open", "high", "low", "close"]:
        out[f"adj_{col}"] = out[col] * factor
    return out
