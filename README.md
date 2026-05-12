# sales-plan

EC事業向けの需要予測ベース販売計画CSV生成アプリです。  
Google Sheetsをデータ管理DBとして参照し、6カ月先の必要数量を算出して、基幹取込用のCSV形式に整形します。

## 目的

- 6カ月後に「何が何点必要か」を出す
- 根拠が過去実績に基づくかを判定する
- 根拠が弱いSKUを人為判断対象として抽出する
- 最終出力CSVを固定フォーマットで生成する
- SKU・カテゴリ・根拠状態を3Dノード球体として直感的に可視化する

## 本番データ構成

本番では `data/*.csv` は主データとして使いません。  
Googleスプレッドシートをデータ管理DBとして使い、Streamlitが直接読み込みます。

```text
Google Spreadsheet
  ├─ product_master
  ├─ sales_history
  ├─ inventory
  └─ overrides
        ↓
Streamlit / src.io.load_google_sheets()
        ↓
src.forecast.forecast()
        ↓
3Dノード可視化 / CSV出力
