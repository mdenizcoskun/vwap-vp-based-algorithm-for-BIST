"""Borsa Istanbul universe discovery and normal-share filtering."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import unicodedata

import pandas as pd
import requests

from .config import BIST_UNIVERSE, BacktestConfig


def to_epoch(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text.upper().strip()


def normalize_bist_symbol(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    symbol = str(value).strip().upper()
    if not symbol:
        return None
    symbol = symbol.replace("BIST:", "").replace("BIST/", "").replace(" ", "")
    symbol = symbol.replace("_IS", ".IS")
    if "." not in symbol:
        symbol = f"{symbol}.IS"
    return symbol


def is_truthy(value: object) -> bool:
    return normalize_text(value) in {"1", "TRUE", "T", "YES", "Y", "EVET", "E"}


def is_falsey(value: object) -> bool:
    return normalize_text(value) in {"0", "FALSE", "F", "NO", "N", "HAYIR", "H"}


def is_normal_share_row(row: pd.Series, normal_share_only: bool = True) -> bool:
    if not normal_share_only:
        return True

    for col in ["is_normal_share", "normal_share", "is_equity", "include"]:
        if col in row.index:
            if is_falsey(row[col]):
                return False
            if is_truthy(row[col]):
                return True

    metadata_cols = [
        "instrument_type",
        "instrument",
        "security_type",
        "asset_type",
        "type",
        "category",
        "market",
        "pazar",
        "segment",
        "group",
        "grup",
        "name",
        "description",
        "sector",
        "industry",
    ]
    bad_terms = [
        "VARANT",
        "WARRANT",
        "SERTIFIKA",
        "CERTIFICATE",
        "FON",
        "FUND",
        "ETF",
        "BYF",
        "BORSA YATIRIM FONU",
        "RUCHAN",
        "RIGHT",
        "KUPON",
        "YENI PAY ALMA",
        "VIOP",
        "FUTURES",
        "OPTION",
        "OPSİYON",
        "OPSIYON",
    ]
    good_terms = ["PAY", "HISSE", "HISSE SENEDI", "EQUITY", "STOCK"]
    metadata_values = [normalize_text(row[col]) for col in metadata_cols if col in row.index and not pd.isna(row[col])]
    joined = " ".join(metadata_values)
    if any(term in joined for term in bad_terms):
        return False
    if joined and any(term in joined for term in good_terms):
        return True
    return True


def load_universe_symbols(cfg: BacktestConfig) -> List[str]:
    if cfg.universe_source == "tradingview":
        universe = fetch_tradingview_universe(cfg)
        symbols = universe["symbol"].dropna().sort_values().unique().tolist()
        if not symbols:
            raise ValueError("TradingView universe returned no valid symbols.")
        return symbols

    if cfg.universe_file is None:
        return sorted(set(BIST_UNIVERSE))

    path = Path(cfg.universe_file)
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        universe = pd.read_excel(path)
    else:
        universe = pd.read_csv(path)

    universe.columns = [str(c).strip().lower() for c in universe.columns]
    symbol_col = next((c for c in ["symbol", "ticker", "code", "kod", "pay_kodu", "hisse", "tv_symbol"] if c in universe.columns), None)
    if symbol_col is None:
        raise ValueError("Universe file must contain one of these columns: symbol, ticker, code, kod, pay_kodu, hisse, tv_symbol")

    rows = []
    for _, row in universe.iterrows():
        if not is_normal_share_row(row, cfg.normal_share_only):
            continue
        symbol = normalize_bist_symbol(row[symbol_col])
        if symbol:
            rows.append(symbol)

    symbols = sorted(set(rows))
    if not symbols:
        raise ValueError(f"No valid symbols found in universe file: {path}")
    return symbols


def fetch_tradingview_universe(cfg: BacktestConfig) -> pd.DataFrame:
    url = "https://scanner.tradingview.com/turkey/scan"
    columns = [
        "name",
        "description",
        "type",
        "subtype",
        "exchange",
        "market",
        "sector",
        "industry",
        "close",
        "volume",
        "market_cap_basic",
    ]
    payload = {
        "columns": columns,
        "filter": [{"left": "type", "operation": "equal", "right": "stock"}],
        "ignore_unknown_fields": False,
        "options": {"lang": "tr"},
        "range": [0, 2000],
        "sort": {"sortBy": "name", "sortOrder": "asc"},
        "symbols": {},
        "markets": ["turkey"],
    }
    response = requests.post(
        url,
        json=payload,
        headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    payload_out = response.json()
    rows = []
    for item in payload_out.get("data", []):
        data = item.get("d", [])
        record = {col: data[idx] if idx < len(data) else None for idx, col in enumerate(columns)}
        record["tv_symbol"] = item.get("s")
        record["symbol"] = normalize_bist_symbol(record.get("name"))
        record["instrument_type"] = record.get("type")
        rows.append(record)

    universe = pd.DataFrame(rows)
    if universe.empty:
        return universe
    if cfg.normal_share_only:
        universe = universe[universe.apply(lambda row: is_normal_share_row(row, True), axis=1)]
    universe = universe.dropna(subset=["symbol"]).drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    return universe
