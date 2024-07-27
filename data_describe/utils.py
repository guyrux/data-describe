from typing import Union

import numpy as np
import pandas as pd


def clean_data(
    df: pd.DataFrame,
    subset: Union[list[str], str],
    cleaning_type: int = 1,
    verbose: int = 0,
):
    """Duplicated registries, null values and negative or zero removing."""
    df_cleaned = df.drop_duplicates()
    df_cleaned.dropna(subset=subset, inplace=True)

    if cleaning_type == 1:
        for column in subset:
            df_cleaned = df_cleaned.loc[df_cleaned[column] > 0, :].copy()

    elif cleaning_type == 2:
        for column in subset:
            df_cleaned = df_cleaned.loc[df_cleaned[column] != 0, :].copy()

    if verbose != 0:
        print(
            f"""Original length: {df.shape[0]} registries
Length after cleaning: {df_cleaned.shape[0]} registries"""
        )

    return df_cleaned


def remove_outliers(
    df: pd.DataFrame,
    lst_columns: list,
    low_boundary_percentile: float = 1,
    up_boundary_percentile: float = 99,
) -> pd.DataFrame:
    """Remove outliers according to a low and up boundaries.

    Args:
        df (pd.DataFrame)
        lst_columns (list)
        low_boundary_percentile (float, optional). Defaults to 1.
        up_boundary_percentile (float, optional). Defaults to 99.

    Returns:
        pd.DataFrame: pd.DataFrame without outliers.
    """    
    df_temp = df.copy()

    for column in lst_columns:
        lb = np.nanpercentile(df_temp[column], low_boundary_percentile)
        ub = np.nanpercentile(df_temp[column], up_boundary_percentile)
        df_temp = df_temp.loc[
            (df_temp[column] >= lb) & (df_temp[column] <= ub), :
        ].copy()

    print(
        f"""Lines removed: {100 * (1 - df_temp.shape[0] / df.shape[0]):.2f}%"""
    )

    return df_temp
