import pandas as pd
from typing import List

def summarize_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Compute descriptive statistics for numeric columns."""
    if not numeric_cols:
        return pd.DataFrame(
            columns=[
                "column",
                "count",
                "mean",
                "std",
                "min",
                "p25",
                "median",
                "p75",
                "max",
            ]
        )

    # BLANK 6: Create a transposed describe table with percentiles 0.25, 0.5, 0.75
    # HINT: df[numeric_cols].describe(...).T
    summary = df[numeric_cols].describe(percentiles = [0.25, 0.5, 0.75]).T

    summary = summary.rename(columns={"50%": "median", "25%": "p25", "75%": "p75"})
    summary.insert(0, "column", summary.index)
    summary.reset_index(drop=True, inplace=True)
    return summary


def summarize_categorical(
    df: pd.DataFrame, cat_cols: List[str], top_k: int = 10
) -> pd.DataFrame:
    """Compute descriptive statistics for categorical columns."""
    rows = []
    for c in cat_cols:
        series = df[c].astype("string")
        n = int(series.shape[0])
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))

        # BLANK 7: top_k value counts (drop missing)
        top = series.dropna().value_counts().head(top_k)

        rows.append(
            {
                "column": c,
                "count": n,
                "missing": n_missing,
                "unique": n_unique,
                "top_values": "; ".join([f"{idx} ({val})" for idx, val in top.items()]),
            }
        )
    return pd.DataFrame(rows)

def correlations(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Compute correlations for numeric columns."""
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    # BLANK 8: compute correlation matrix for numeric columns
    corr = df[numeric_cols].corr()
    return corr

def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    (Student task): Create a missingness table.

    Requirements:
    - Compute missing_rate for each column (fraction missing)
    - Compute missing_count for each column
    - Return a DataFrame with columns:
        column, missing_rate, missing_count
    - Sort by missing_rate descending

    Hints:
    - df.isna().mean() gives missing rates
    - df.isna().sum() gives missing counts
    """
    missing_rate = df.isna().mean()
    missing_count = df.isna().sum()
    miss_df = pd.DataFrame({
        "column": missing_rate.index,
        "missing_rate": missing_rate.values,
        "missing_count": missing_count.values,
    })
    miss_df = miss_df.sort_values("missing_rate", ascending = False).reset_index(drop = True)
    return miss_df