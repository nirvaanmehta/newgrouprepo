import pandas as pd
from typing import Tuple, List

def basic_profile(df: pd.DataFrame) -> dict:
    """Return a basic JSON-serializable profile of the dataset."""
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        # BLANK 4: list of column names
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "n_missing_total": int(df.isna().sum().sum()),
        "missing_by_col": df.isna().sum().to_dict(),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
    }


def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify and split numeric vs categorical columns into numeric and categorical lists."""
    # BLANK 5: list numeric column names
    # HINT: df.select_dtypes(include=["number"]).columns.______
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Treat everything else as categorical
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, cat_cols
