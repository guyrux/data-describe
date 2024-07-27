# %%
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_describe.utils import clean_data


def describing_continuous_data(
    df: pd.DataFrame, lst_columns: Union[list[str], str]
):
    """Describe a pd.DataFrame in each column within a given list.

    Args:
        df (pd.DataFrame):
        lst_columns (Union[list[str], str]):

    Returns:
        _type_: Return a pd.DataFrame with the describe of each column.
    """
    dct_temp = {}
    df_temp = pd.DataFrame()

    for column in lst_columns:
        dct_temp[column] = clean_data(df=df, subset=[column], verbose=0)[
            [column]
        ]
        df_temp = pd.concat(
            [df_temp, dct_temp[column].describe().T.reset_index()], axis=0
        )

    return df_temp


def plot_histogram(
    df: pd.DataFrame, column: str, pivot: float = np.nan
) -> None:
    """Plot a histogram with a kde and a vertical line for a given pivot value.

    Args:
        df (pd.DataFrame)
        column (str)
        pivot (float, optional): It's a value of a vertical. Defaults to nan.

    Returns:
        _type_: _description_
    """
    plt.figure(figsize=(6, 2))
    sns.histplot(data=df, x=column, bins=50, kde=True)
    plt.axvline(x=pivot, color='red', linestyle='--', linewidth=1)
    text = f'Pivot: {pivot:.2f}'
    plt.text(
        pivot,
        plt.gca().get_ylim()[1],
        f'{text}',
        horizontalalignment='right',
        verticalalignment='bottom',
        color='red',
    )
    plt.show()
    return None
