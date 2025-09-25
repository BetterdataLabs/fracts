import pandas as pd


def extract(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df[[c for c in df.columns if c.startswith(f"{prefix}.")]]
    df = df.set_axis([c[len(prefix) + 1:] for c in df.columns], axis=1)
    return df


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.nlevels > 1:
        df = df.set_axis([".".join(c) for c in df.columns], axis=1)
    return df