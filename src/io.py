from __future__ import annotations

import io
from typing import Dict

import pandas as pd


def read_csv_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    raw = uploaded_file.getvalue()
    for enc in ["utf-8-sig", "cp932", "utf-8"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw))


def to_csv_bytes(df: pd.DataFrame, encoding: str = "cp932") -> bytes:
    return df.to_csv(index=False).encode(encoding, errors="replace")


def load_sample_data() -> Dict[str, pd.DataFrame]:
    return {
        "product_master": pd.read_csv("data/product_master.csv"),
        "sales_history": pd.read_csv("data/sales_history.csv"),
        "inventory": pd.read_csv("data/inventory.csv"),
        "overrides": pd.read_csv("data/overrides.csv"),
    }
