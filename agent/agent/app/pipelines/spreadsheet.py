from pathlib import Path
import pandas as pd


def extract_structured_data_from_spreadsheet(file_path: Path) -> dict:
    ext = file_path.suffix.lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext == ".xlsx":
            df = pd.read_excel(file_path, engine="openpyxl")
        elif ext == ".xls":
            df = pd.read_excel(file_path, engine="xlrd")
        else:
            return {
                "status": "error",
                "summary": f"Unsupported spreadsheet type: {ext}",
                "table_info": {},
                "sample_rows": [],
                "null_counts": {},
                "numeric_summary": {},
            }

        df = normalize_dataframe(df)

        return {
            "status": "processed",
            "summary": "Spreadsheet analyzed successfully",
            "table_info": {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "column_names": [str(col) for col in df.columns.tolist()],
            },
            "sample_rows": df.head(5).fillna("").to_dict(orient="records"),
            "null_counts": {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
            "numeric_summary": build_numeric_summary(df),
        }

    except Exception as e:
        return {
            "status": "error",
            "summary": f"Spreadsheet processing failed: {str(e)}",
            "table_info": {},
            "sample_rows": [],
            "null_counts": {},
            "numeric_summary": {},
        }


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def build_numeric_summary(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        return {}

    summary = {}
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()

        if series.empty:
            summary[str(col)] = {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
            }
            continue

        summary[str(col)] = {
            "count": int(series.count()),
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
        }

    return summary