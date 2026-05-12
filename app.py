from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src.forecast import forecast
from src.io import load_google_sheets, load_sample_data, read_csv_upload, to_csv_bytes

st.set_page_config(page_title="Sales Plan | 需要予測見積", layout="wide")

st.title("Sales Plan")
st.caption("6カ月後に何が何点必要かを、過去実績・在庫・人為判断に分解してCSV化する。")

with st.sidebar:
    st.header("Input")
    quote_date = st.date_input("見積日", value=date.today())
    input_source = st.radio(
        "データ取得元",
        ["Google Sheets", "サンプルCSV", "CSVアップロード"],
        index=0,
        help="本番運用はGoogle Sheets。サンプルCSVは検証用、CSVアップロードは緊急時の手動確認用。",
    )
    st.divider()
    st.caption("Google Sheets利用時は `.streamlit/secrets.toml` に認証情報と Spreadsheet ID を設定します。")

try:
    if input_source == "Google Sheets":
        data = load_google_sheets(st.secrets)
    elif input_source == "サンプルCSV":
        data = load_sample_data()
    else:
        c1, c2 = st.columns(2)
        with c1:
            product_master_file = st.file_uploader("product_master.csv", type="csv")
            sales_history_file = st.file_uploader("sales_history.csv", type="csv")
        with c2:
            inventory_file = st.file_uploader("inventory.csv", type="csv")
            overrides_file = st.file_uploader("overrides.csv", type="csv")
        data = {
            "product_master": read_csv_upload(product_master_file),
            "sales_history": read_csv_upload(sales_history_file),
            "inventory": read_csv_upload(inventory_file),
            "overrides": read_csv_upload(overrides_file),
        }
except Exception as e:
    st.error("データ取得に失敗しました。")
    st.exception(e)
    st.stop()

missing = [k for k in ["product_master", "sales_history"] if data.get(k, pd.DataFrame()).empty]
if missing:
    st.warning(f"必須データが未投入です: {', '.join(missing)}")
    st.stop()

try:
    output, review = forecast(
        data["product_master"],
        data["sales_history"],
        data.get("inventory"),
        data.get("overrides"),
        pd.Timestamp(quote_date),
    )
except Exception as e:
    st.error("計算に失敗しました。列名・日付・数量の型を確認してください。")
    st.exception(e)
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("CSV出力SKU", f"{len(output):,}")
k2.metric("推奨数量合計", f"{int(output['数量'].sum()) if len(output) else 0:,}")
k3.metric("推奨金額", f"¥{int(output['金額'].sum()) if len(output) else 0:,}")
k4.metric("要確認SKU", f"{int(review['要確認'].sum()) if len(review) else 0:,}")

tab1, tab2, tab3, tab4 = st.tabs(["CSV出力", "要確認", "根拠", "入力データ"])

with tab1:
    st.subheader("最終出力CSV")
    st.dataframe(output, use_container_width=True, hide_index=True)
    st.download_button(
        "CSVをダウンロード",
        data=to_csv_bytes(output),
        file_name=f"sales_plan_{pd.Timestamp(quote_date).strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

with tab2:
    st.subheader("根拠が弱い・人為判断が必要なSKU")
    st.caption("C/D判定、直近販売なし、手動上書き、廃番除外を上位に出す。ここが運用上の判断レイヤー。")
    st.dataframe(review[review["要確認"] | (review["異常メモ"].astype(str) != "")], use_container_width=True, hide_index=True)

with tab3:
    st.subheader("計算根拠テーブル")
    st.dataframe(review, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("投入データ")
    for name, df in data.items():
        with st.expander(name, expanded=False):
            st.dataframe(df, use_container_width=True, hide_index=True)
