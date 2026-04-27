import pandas as pd
import pathlib as Path 

def ensure_dirs(reports: Path) -> None:
    """Create output folders."""
    # BLANK 1: create the figures folder
    # HINT: (reports / "figures").mkdir(...)
    (reports/"figures").mkdir (parents = True, exist_ok = True)


def read_data(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame with basic error handling."""
    # BLANK 2: raise FileNotFoundError if path does not exist
    if not path.exists(): 
        raise FileNotFoundError(f"file not found: {path}")

    # BLANK 3: read the CSV into df
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataframe is empty.")
    return df