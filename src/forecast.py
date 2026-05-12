from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .config import DEFAULTS, OUTPUT_COLUMNS


def _to_month_start(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()


def _ceil_to_lot(qty: float, lot_size: float, moq: float) -> int:
    if pd.isna(qty) or qty <= 0:
        return 0
    lot = int(lot_size) if not pd.isna(lot_size) and lot_size > 0 else 1
    minimum = int(moq) if not pd.isna(moq) and moq > 0 else 0
    rounded = int(math.ceil(qty / lot) * lot)
    return max(rounded, minimum) if rounded > 0 else 0


def _safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def normalize_inputs(
    product_master: pd.DataFrame,
    sales_history: pd.DataFrame,
    inventory: pd.DataFrame | None = None,
    overrides: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pm = product_master.copy()
    sales = sales_history.copy()
    inv = inventory.copy() if inventory is not None and not inventory.empty else pd.DataFrame(columns=["商品コード", "現在庫", "入荷予定", "引当済"])
    ov = overrides.copy() if overrides is not None and not overrides.empty else pd.DataFrame(columns=["商品コード", "上書き数量", "除外フラグ", "理由"])

    for df in [pm, sales, inv, ov]:
        if "商品コード" in df.columns:
            df["商品コード"] = df["商品コード"].astype(str).str.strip()

    pm["単価"] = _safe_num(pm, "単価", 0)
    pm["安全在庫日数"] = _safe_num(pm, "安全在庫日数", DEFAULTS["safety_stock_days"])
    pm["発注ロット"] = _safe_num(pm, "発注ロット", DEFAULTS["lot_size"])
    pm["最小発注数"] = _safe_num(pm, "最小発注数", DEFAULTS["moq"])
    pm["廃番フラグ"] = pm.get("廃番フラグ", "").astype(str).str.upper().isin(["1", "TRUE", "YES", "Y", "廃番"])

    sales["日付"] = pd.to_datetime(sales["日付"], errors="coerce")
    sales["月"] = _to_month_start(sales["日付"])
    sales["数量"] = _safe_num(sales, "数量", 0)
    sales = sales.dropna(subset=["月", "商品コード"])

    for c in ["現在庫", "入荷予定", "引当済"]:
        inv[c] = _safe_num(inv, c, 0)

    ov["上書き数量"] = _safe_num(ov, "上書き数量", np.nan)
    ov["除外フラグ"] = ov.get("除外フラグ", "").astype(str).str.upper().isin(["1", "TRUE", "YES", "Y", "除外"])
    ov["理由"] = ov.get("理由", "").fillna("").astype(str)
    return pm, sales, inv, ov


def build_monthly_panel(sales: pd.DataFrame, product_master: pd.DataFrame, quote_date: pd.Timestamp) -> pd.DataFrame:
    start = sales["月"].min()
    if pd.isna(start):
        start = (quote_date - pd.DateOffset(months=24)).to_period("M").to_timestamp()
    end = quote_date.to_period("M").to_timestamp() - pd.DateOffset(months=1)
    months = pd.date_range(start=start, end=end, freq="MS")
    skus = product_master["商品コード"].dropna().astype(str).unique()
    idx = pd.MultiIndex.from_product([skus, months], names=["商品コード", "月"])
    monthly = sales.groupby(["商品コード", "月"], as_index=True)["数量"].sum().reindex(idx, fill_value=0).reset_index()
    return monthly


def category_seasonality(monthly: pd.DataFrame, product_master: pd.DataFrame) -> pd.DataFrame:
    base = monthly.merge(product_master[["商品コード", "カテゴリ"]], on="商品コード", how="left")
    cat_month = base.groupby(["カテゴリ", base["月"].dt.month])["数量"].mean().rename("cat_month_avg").reset_index()
    cat_avg = base.groupby("カテゴリ")["数量"].mean().rename("cat_avg").reset_index()
    out = cat_month.merge(cat_avg, on="カテゴリ", how="left")
    out["cat_season_idx"] = np.where(out["cat_avg"] > 0, out["cat_month_avg"] / out["cat_avg"], 1.0)
    out["cat_season_idx"] = out["cat_season_idx"].clip(0.55, 1.8).fillna(1.0)
    out = out.rename(columns={"月": "月番号"})[["カテゴリ", "月番号", "cat_season_idx"]]
    return out


def _sku_stats(g: pd.DataFrame, quote_date: pd.Timestamp) -> pd.Series:
    g = g.sort_values("月")
    q_month = quote_date.to_period("M").to_timestamp()
    last_3 = g[g["月"].between(q_month - pd.DateOffset(months=3), q_month - pd.DateOffset(months=1))]["数量"]
    last_6 = g[g["月"].between(q_month - pd.DateOffset(months=6), q_month - pd.DateOffset(months=1))]["数量"]
    last_12 = g[g["月"].between(q_month - pd.DateOffset(months=12), q_month - pd.DateOffset(months=1))]["数量"]
    months_with_sales = int((g["数量"] > 0).sum())
    total_months = int(len(g))
    x = np.arange(len(g[-12:]))
    y = g.tail(12)["数量"].to_numpy(dtype=float)
    slope = 0.0
    if len(y) >= 4 and np.nanstd(y) > 0:
        slope = float(np.polyfit(x, y, 1)[0])
    trend_ratio = 1.0
    denom = max(float(np.nanmean(y)), 1.0)
    trend_ratio = float(np.clip(1 + slope / denom, 0.70, 1.35))
    recent_avg = float(last_3.mean()) if len(last_3) else 0.0
    mid_avg = float(last_6.mean()) if len(last_6) else 0.0
    stable_avg = float(last_12.mean()) if len(last_12) else 0.0
    return pd.Series({
        "直近3M平均": recent_avg,
        "直近6M平均": mid_avg,
        "直近12M平均": stable_avg,
        "販売月数": months_with_sales,
        "履歴月数": total_months,
        "トレンド係数": trend_ratio,
        "直近6M販売数": float(last_6.sum()) if len(last_6) else 0.0,
    })


def forecast(product_master: pd.DataFrame, sales_history: pd.DataFrame, inventory: pd.DataFrame | None, overrides: pd.DataFrame | None, quote_date: str | pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    quote_date = pd.to_datetime(quote_date)
    pm, sales, inv, ov = normalize_inputs(product_master, sales_history, inventory, overrides)
    monthly = build_monthly_panel(sales, pm, quote_date)
    cat_season = category_seasonality(monthly, pm)

    stats = monthly.groupby("商品コード", group_keys=False).apply(lambda g: _sku_stats(g, quote_date)).reset_index()
    base = pm.merge(stats, on="商品コード", how="left")

    # Base monthly demand: recent actuals have priority, stable demand prevents overreaction.
    base["月次基準需要"] = (
        base["直近3M平均"].fillna(0) * 0.50 +
        base["直近6M平均"].fillna(0) * 0.30 +
        base["直近12M平均"].fillna(0) * 0.20
    )
    base["月次基準需要"] = base["月次基準需要"].clip(lower=0)

    horizon_months = pd.date_range(
        start=quote_date.to_period("M").to_timestamp() + pd.DateOffset(months=1),
        periods=6,
        freq="MS",
    )
    rows = []
    for _, r in base.iterrows():
        demand = 0.0
        for m in horizon_months:
            season = cat_season.loc[
                (cat_season["カテゴリ"] == r.get("カテゴリ")) & (cat_season["月番号"] == m.month),
                "cat_season_idx",
            ]
            season_idx = float(season.iloc[0]) if len(season) else 1.0
            demand += float(r.get("月次基準需要", 0) or 0) * float(r.get("トレンド係数", 1) or 1) * season_idx
        rows.append((r["商品コード"], demand))
    horizon = pd.DataFrame(rows, columns=["商品コード", "6カ月予測需要"])
    base = base.merge(horizon, on="商品コード", how="left")

    inv_agg = inv.groupby("商品コード", as_index=False)[["現在庫", "入荷予定", "引当済"]].sum()
    base = base.merge(inv_agg, on="商品コード", how="left")
    for c in ["現在庫", "入荷予定", "引当済"]:
        base[c] = base[c].fillna(0)
    base["有効在庫"] = base["現在庫"] + base["入荷予定"] - base["引当済"]
    base["安全在庫数量"] = (base["月次基準需要"] / 30.0 * base["安全在庫日数"]).fillna(0)
    base["不足数量_raw"] = (base["6カ月予測需要"] + base["安全在庫数量"] - base["有効在庫"]).clip(lower=0)
    base["推奨数量"] = base.apply(lambda r: _ceil_to_lot(r["不足数量_raw"], r["発注ロット"], r["最小発注数"]), axis=1)

    # Evidence grading: separates historically-grounded items from human-review candidates.
    base["根拠レベル"] = np.select(
        [
            (base["履歴月数"] >= 12) & (base["販売月数"] >= 6) & (base["直近6M販売数"] > 0),
            (base["履歴月数"] >= 6) & (base["販売月数"] >= 2),
            (base["販売月数"] >= 1),
        ],
        ["A:実績根拠あり", "B:限定実績", "C:弱い実績"],
        default="D:要人為判断",
    )
    base["要確認"] = base["根拠レベル"].isin(["C:弱い実績", "D:要人為判断"])
    base["異常メモ"] = ""
    base.loc[(base["推奨数量"] > 0) & (base["直近6M販売数"] == 0), "異常メモ"] = "直近販売なし。カテゴリ季節性/手動根拠確認"
    base.loc[base["廃番フラグ"], "異常メモ"] = "廃番除外"
    base.loc[base["廃番フラグ"], "推奨数量"] = 0

    base = base.merge(ov[["商品コード", "上書き数量", "除外フラグ", "理由"]], on="商品コード", how="left")
    base.loc[base["除外フラグ"].fillna(False), "推奨数量"] = 0
    has_manual = base["上書き数量"].notna()
    base.loc[has_manual, "推奨数量"] = base.loc[has_manual, "上書き数量"].clip(lower=0).round().astype(int)
    base.loc[has_manual, "異常メモ"] = "手動上書き: " + base.loc[has_manual, "理由"].fillna("").astype(str)

    result = base[base["推奨数量"] > 0].copy()
    for col, default in [
        ("得意先コード", DEFAULTS["得意先コード"]),
        ("納入期限コード", DEFAULTS["納入期限コード"]),
        ("担当コード", DEFAULTS["担当コード"]),
        ("倉庫コード", DEFAULTS["倉庫コード"]),
        ("色", ""),
        ("サイズ", ""),
    ]:
        if col not in result.columns:
            result[col] = default
        result[col] = result[col].fillna(default).astype(str)
        if col in ["得意先コード", "担当コード", "倉庫コード"]:
            result[col] = result[col].str.replace(r"\.0$", "", regex=True)
        if col == "得意先コード":
            result[col] = result[col].str.zfill(4)

    result["カテゴリ"] = result["カテゴリ"].fillna("")
    result["見積日"] = quote_date.strftime("%Y/%-m/%-d") if hasattr(quote_date, "strftime") else str(quote_date)
    result["件名"] = DEFAULTS["件名"]
    result["摘要"] = DEFAULTS["摘要"]
    result["数量"] = result["推奨数量"].astype(int)
    result["単価"] = result["単価"].fillna(0).round(0).astype(int)
    result["金額"] = result["数量"] * result["単価"]
    result["備考"] = (
        result["根拠レベル"].astype(str) + " / 6M予測=" + result["6カ月予測需要"].round(1).astype(str) +
        " / 有効在庫=" + result["有効在庫"].round(0).astype(int).astype(str)
    )
    result.loc[result["異常メモ"].astype(str) != "", "備考"] += " / " + result["異常メモ"].astype(str)

    output = result[OUTPUT_COLUMNS].sort_values(["カテゴリ", "商品コード"]).reset_index(drop=True)
    review = base.sort_values(["要確認", "推奨数量", "6カ月予測需要"], ascending=[False, False, False])[
        ["商品コード", "カテゴリ", "推奨数量", "6カ月予測需要", "有効在庫", "根拠レベル", "要確認", "異常メモ", "直近3M平均", "直近6M平均", "直近12M平均", "トレンド係数"]
    ].reset_index(drop=True)
    return output, review
