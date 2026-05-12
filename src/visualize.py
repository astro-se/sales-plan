from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


STATUS_COLOR_MAP = {
    "A": "#35C759",          # strong
    "B": "#4A90E2",          # normal
    "C": "#F5A623",          # review
    "D": "#D0021B",          # alert
    "手動上書き": "#8E44AD",  # override
    "推奨なし": "#8A8A8A",
    "不明": "#8A8A8A",
}


def fibonacci_sphere(n: int, radius: float = 1.0) -> np.ndarray:
    """
    Place n points on a sphere using the Fibonacci sphere algorithm.
    This produces a clean, non-grid-like distribution for node visualization.
    """
    if n <= 0:
        return np.empty((0, 3))

    if n == 1:
        return np.array([[0.0, 0.0, radius]])

    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(n):
        y = 1.0 - (i / float(n - 1)) * 2.0
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i

        x = np.cos(theta) * r
        z = np.sin(theta) * r
        points.append((radius * x, radius * y, radius * z))

    return np.array(points)


def _safe_columns(df: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    """
    Ensure required columns exist.
    This keeps the visualization layer tolerant of calculation-layer schema changes.
    """
    result = df.copy()
    for col in required:
        if col not in result.columns:
            result[col] = ""
    return result


def _normalize_status(row: pd.Series) -> str:
    """
    Decide the visual status for a SKU node.

    Priority:
    1. Manual override
    2. No recommendation
    3. Evidence rank
    """
    override_text = str(row.get("手動上書き", "")).strip()
    if override_text in ["TRUE", "True", "true", "1", "あり", "有", "上書き"]:
        return "手動上書き"

    quantity = pd.to_numeric(row.get("推奨数量", row.get("数量", 0)), errors="coerce")
    if pd.isna(quantity) or quantity <= 0:
        return "推奨なし"

    rank = str(row.get("根拠レベル", row.get("根拠ランク", row.get("evidence_rank", "")))).strip()
    if rank in STATUS_COLOR_MAP:
        return rank

    if bool(row.get("要確認", False)):
        return "D"

    return "不明"


def _merge_visual_data(review: pd.DataFrame, output: pd.DataFrame) -> pd.DataFrame:
    """
    Merge forecast review table and final output table into a visualization-friendly table.
    The app's calculation layer may keep evidence metrics in review and final CSV fields in output.
    """
    review = review.copy() if review is not None else pd.DataFrame()
    output = output.copy() if output is not None else pd.DataFrame()

    review = _safe_columns(
        review,
        [
            "商品コード",
            "カテゴリ",
            "推奨数量",
            "根拠レベル",
            "要確認",
            "異常メモ",
            "手動上書き",
            "予測6カ月需要",
            "有効在庫",
            "安全在庫",
        ],
    )

    output = _safe_columns(
        output,
        [
            "商品コード",
            "カテゴリ",
            "数量",
            "単価",
            "金額",
            "備考",
        ],
    )

    if review.empty and output.empty:
        return pd.DataFrame()

    if review.empty:
        df = output.copy()
        df["推奨数量"] = pd.to_numeric(df["数量"], errors="coerce").fillna(0)
        df["根拠レベル"] = "不明"
        df["要確認"] = False
        df["異常メモ"] = ""
        df["手動上書き"] = ""
        return df

    if output.empty:
        df = review.copy()
        if "推奨数量" not in df.columns:
            df["推奨数量"] = 0
        df["数量"] = df["推奨数量"]
        df["単価"] = 0
        df["金額"] = 0
        df["備考"] = df.get("異常メモ", "")
        return df

    output_small = output[
        [
            "商品コード",
            "数量",
            "単価",
            "金額",
            "備考",
        ]
    ].copy()

    df = review.merge(output_small, on="商品コード", how="left")

    if "推奨数量" not in df.columns:
        df["推奨数量"] = df["数量"]

    df["推奨数量"] = pd.to_numeric(df["推奨数量"], errors="coerce")
    df["数量"] = pd.to_numeric(df["数量"], errors="coerce")

    df["推奨数量"] = df["推奨数量"].fillna(df["数量"]).fillna(0)
    df["数量"] = df["数量"].fillna(df["推奨数量"]).fillna(0)

    return df


def _filter_visual_data(
    df: pd.DataFrame,
    category_filter: list[str] | None,
    status_filter: list[str] | None,
    quantity_min: int,
    max_nodes: int,
) -> pd.DataFrame:
    if df.empty:
        return df

    result = df.copy()
    result["推奨数量"] = pd.to_numeric(result["推奨数量"], errors="coerce").fillna(0)
    result["visual_status"] = result.apply(_normalize_status, axis=1)

    if category_filter:
        result = result[result["カテゴリ"].astype(str).isin(category_filter)]

    if status_filter:
        result = result[result["visual_status"].astype(str).isin(status_filter)]

    result = result[result["推奨数量"] >= quantity_min]

    result = result.sort_values(
        by=["推奨数量", "商品コード"],
        ascending=[False, True],
    )

    if max_nodes > 0:
        result = result.head(max_nodes)

    return result.reset_index(drop=True)


def build_node_sphere_figure(
    review: pd.DataFrame,
    output: pd.DataFrame,
    category_filter: list[str] | None = None,
    status_filter: list[str] | None = None,
    quantity_min: int = 0,
    max_nodes: int = 300,
) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Build a 3D spherical node visualization for Streamlit.

    Structure:
    - Center node: Sales Plan / Forecast Engine
    - Inner sphere: categories
    - Outer sphere: SKU nodes
    - Lines: center -> category -> SKU

    Visual semantics:
    - SKU node size = recommended quantity
    - SKU node color = evidence / review status
    """
    merged = _merge_visual_data(review, output)
    df = _filter_visual_data(
        merged,
        category_filter=category_filter,
        status_filter=status_filter,
        quantity_min=quantity_min,
        max_nodes=max_nodes,
    )

    fig = go.Figure()

    fig.update_layout(
        paper_bgcolor="#070707",
        plot_bgcolor="#070707",
        margin=dict(l=0, r=0, t=0, b=0),
        height=760,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EDEDED", size=11),
        ),
        scene=dict(
            bgcolor="#070707",
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            camera=dict(eye=dict(x=1.35, y=1.35, z=1.05)),
            aspectmode="cube",
        ),
    )

    # Empty state
    if df.empty:
        fig.add_annotation(
            text="表示対象のノードがありません",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=18, color="#FFFFFF"),
        )
        return fig, df

    categories = sorted(df["カテゴリ"].astype(str).fillna("未分類").unique().tolist())

    category_coords = fibonacci_sphere(len(categories), radius=4.2)
    category_position = {
        category: category_coords[i]
        for i, category in enumerate(categories)
    }

    sku_coords = fibonacci_sphere(len(df), radius=9.5)

    df["x"] = sku_coords[:, 0]
    df["y"] = sku_coords[:, 1]
    df["z"] = sku_coords[:, 2]

    df["visual_status"] = df.apply(_normalize_status, axis=1)
    df["visual_color"] = df["visual_status"].map(STATUS_COLOR_MAP).fillna("#8A8A8A")

    max_qty = max(float(df["推奨数量"].max()), 1.0)
    df["visual_size"] = 7.0 + 22.0 * np.sqrt(df["推奨数量"].clip(lower=0) / max_qty)

    # Center node
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers+text",
            marker=dict(
                size=20,
                color="#FFFFFF",
                opacity=1,
                line=dict(color="#D6D6D6", width=1.5),
            ),
            text=["Sales Plan"],
            textposition="top center",
            textfont=dict(color="#FFFFFF", size=13),
            hovertext="Forecast Engine<br>6カ月先の需要予測・根拠判定・CSV出力",
            hoverinfo="text",
            name="Forecast Engine",
        )
    )

    # Category nodes
    for category in categories:
        x, y, z = category_position[category]
        count = int((df["カテゴリ"].astype(str) == category).sum())
        qty_sum = int(df.loc[df["カテゴリ"].astype(str) == category, "推奨数量"].sum())

        fig.add_trace(
            go.Scatter3d(
                x=[x],
                y=[y],
                z=[z],
                mode="markers+text",
                marker=dict(
                    size=14 + min(count, 30) * 0.25,
                    color="#FFFFFF",
                    opacity=0.92,
                    line=dict(color="#8C8C8C", width=1),
                ),
                text=[category],
                textposition="top center",
                textfont=dict(color="#FFFFFF", size=10),
                hovertext=(
                    f"カテゴリ: {category}<br>"
                    f"SKU数: {count:,}<br>"
                    f"推奨数量合計: {qty_sum:,}"
                ),
                hoverinfo="text",
                name=f"カテゴリ: {category}",
                showlegend=False,
            )
        )

    # Center -> category lines
    for category in categories:
        x, y, z = category_position[category]
        fig.add_trace(
            go.Scatter3d(
                x=[0, x],
                y=[0, y],
                z=[0, z],
                mode="lines",
                line=dict(width=2, color="rgba(255,255,255,0.18)"),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Category -> SKU lines
    for row in df.itertuples(index=False):
        category = str(getattr(row, "カテゴリ"))
        cx, cy, cz = category_position.get(category, np.array([0.0, 0.0, 0.0]))
        fig.add_trace(
            go.Scatter3d(
                x=[cx, getattr(row, "x")],
                y=[cy, getattr(row, "y")],
                z=[cz, getattr(row, "z")],
                mode="lines",
                line=dict(width=1, color="rgba(255,255,255,0.07)"),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # SKU nodes by status, so legend is clean
    for status, color in STATUS_COLOR_MAP.items():
        status_df = df[df["visual_status"] == status]
        if status_df.empty:
            continue

        hovertexts = []
        for r in status_df.itertuples(index=False):
            product_code = getattr(r, "商品コード", "")
            category = getattr(r, "カテゴリ", "")
            quantity = int(getattr(r, "推奨数量", 0))
            evidence = getattr(r, "根拠レベル", "")
            review_flag = getattr(r, "要確認", "")
            memo = getattr(r, "異常メモ", "")
            stock = getattr(r, "有効在庫", "")
            demand = getattr(r, "予測6カ月需要", "")
            safety = getattr(r, "安全在庫", "")
            amount = getattr(r, "金額", "")

            hovertexts.append(
                f"商品コード: {product_code}<br>"
                f"カテゴリ: {category}<br>"
                f"推奨数量: {quantity:,}<br>"
                f"根拠レベル: {evidence}<br>"
                f"状態: {status}<br>"
                f"要確認: {review_flag}<br>"
                f"予測6カ月需要: {demand}<br>"
                f"有効在庫: {stock}<br>"
                f"安全在庫: {safety}<br>"
                f"金額: {amount}<br>"
                f"メモ: {memo}"
            )

        fig.add_trace(
            go.Scatter3d(
                x=status_df["x"],
                y=status_df["y"],
                z=status_df["z"],
                mode="markers",
                marker=dict(
                    size=status_df["visual_size"],
                    color=color,
                    opacity=0.9,
                    line=dict(color="rgba(255,255,255,0.55)", width=0.6),
                ),
                text=hovertexts,
                hoverinfo="text",
                name=status,
            )
        )

    return fig, df
