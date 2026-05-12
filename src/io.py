from __future__ import annotations

import io
from typing import Dict, Iterable

import pandas as pd


SHEET_NAMES = ["product_master", "sales_history", "inventory", "overrides"]


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


def _records_to_df(records: list[dict]) -> pd.DataFrame:
    """Convert Google Sheets records to a dataframe while preserving blank optional sheets."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df = df.dropna(how="all")
    # Google Sheets sometimes returns purely blank columns with empty names.
    df = df.loc[:, [str(c).strip() != "" for c in df.columns]]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_google_sheets(st_secrets, sheet_names: Iterable[str] = SHEET_NAMES) -> Dict[str, pd.DataFrame]:
    """
    Load input tables from Google Sheets using Streamlit secrets.

    Required secrets structure:

    [gcp_service_account]
    type = "service_account"
    project_id = "..."
    private_key_id = "..."
    private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
    client_email = "..."
    client_id = "..."
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url = "..."

    [google_sheets]
    spreadsheet_id = "..."
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError as e:
        raise RuntimeError("Google Sheets連携に必要なライブラリがありません。`pip install -r requirements.txt`を実行してください。") from e

    if "gcp_service_account" not in st_secrets or "google_sheets" not in st_secrets:
        raise RuntimeError("`.streamlit/secrets.toml` に gcp_service_account と google_sheets を設定してください。")

    service_account_info = dict(st_secrets["gcp_service_account"])
    spreadsheet_id = st_secrets["google_sheets"].get("spreadsheet_id", "")
    if not spreadsheet_id:
        raise RuntimeError("`.streamlit/secrets.toml` の [google_sheets].spreadsheet_id が未設定です。")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    credentials = Credentials.from_service_account_info(service_account_info, scopes=scopes)
    client = gspread.authorize(credentials)
    spreadsheet = client.open_by_key(spreadsheet_id)

    data: Dict[str, pd.DataFrame] = {}
    missing_required = []
    for sheet_name in sheet_names:
        try:
            ws = spreadsheet.worksheet(sheet_name)
            data[sheet_name] = _records_to_df(ws.get_all_records())
        except gspread.WorksheetNotFound:
            if sheet_name in ["product_master", "sales_history"]:
                missing_required.append(sheet_name)
            data[sheet_name] = pd.DataFrame()

    if missing_required:
        raise RuntimeError(f"必須シートが見つかりません: {', '.join(missing_required)}")
    return data
