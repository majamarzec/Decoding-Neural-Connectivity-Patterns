import pandas as pd
import numpy as np
from typing import Any, Literal
from pandas.io.formats.style import Styler
import config as c



def get_scattered_chunks(data: pd.DataFrame, 
                         n_chunks: int = 5, 
                         chunk_size: int = 3
                         ) -> pd.DataFrame:

    """
    Reteurns a subsample of equally scattered chunks of rows.

    Parameters
    __________
    data: pd.DataFrame
          Input dataset
    n_chunks: int
          Number of chunks to collect for the subsample
    chunk_size: int
          Number of rows to include in each chunk
    
    Returns
    _______
    pd.DataFrame
       Subsample data
    """
    endpoint = len(data) - chunk_size
    sample_indices = np.linspace(0, endpoint, n_chunks, dtype = int)
    sample_indices = [ind for i in sample_indices for ind in range(i, i+chunk_size)]
    return data.iloc[sample_indices, :]

# helpers for print_ table
###
def highlight_nan(value: Any) -> str:
    """
    Highlight NaN values in grey.

    Parameters
    __________
    value: Any
           Cell value
    
    Returns
    _______
    str
        Cell background color definition
    """
    color = 'white'
    try:
        value = float(value)
    except (ValueError, TypeError):
        pass
    else: 
        color = "grey" if pd.isna(value) else "white"
    finally:
        return f"background-color: {color}" 

def highlight_positives(test_result: Any) -> str:
    """
    Highlight positive of targets in red.

    Parameters
    __________
    test_results: bool
                  Observed test result
    Returns
    _______
    str
        Cell background color definition
    """
    if pd.isna(test_result):
        return "background-color: grey"
    color = "red" if test_result else "white"
    return f"background-color: {color}"

def get_table_styles(header_font_size: int = 12,
                     cell_font_size: int  = 11,
                     ) -> list:
    """
    Creates a table styles definition to be used by the 'set_table_styles()' method.

    References
    __________
    * Pandas' table styles documentation:
      https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#Table-styles

    Parameters
    __________
    header_font_size: int
                      Header text font size in pixels (px)
    cell_font_size: int
                    Cell text font size in pixels (px)
    
    Returns
    _______
    list
        Table styles definition
    """
    heading_properties = [("font-size", f"{header_font_size}px"), ("color", "black")]
    cell_properties = [("font-size", f"{cell_font_size}px"), ("color", "black")]

    return [dict(selector = "th", props = heading_properties),
            dict(selector = "td", props = cell_properties)]

TextAlign = Literal["left", "right", "center", "justify"]
###

def print_table(data: pd.DataFrame,
                header_font_size: int = 12,
                cell_font_size: int = 11,
                text_align: TextAlign = "center"
                ) -> Styler:
    """
    Returns styled representation of the dataframe.

    Parameters
    __________
    data: pd.DataFrame
          Input data
    header_font_size: int
          Header text font size in pixels (px)
    cell_font_size: int
          Cell text font size in pixels (px)
    text_align: str
         Any of the CSS text-align property options, defaults to "center"
        
    Returns
    _______
    Styler
       Styled dataframe representation
    """
    valid_aligns = {"left", "right", "center", "justify"}
    if text_align not in valid_aligns:
        raise ValueError(f"text_align must be one of {valid_aligns}, got {text_align!r}")
    
    table_styles = get_table_styles(header_font_size=header_font_size,
                                    cell_font_size=cell_font_size)
    styler = Styler(data = data, table_styles = table_styles)
    styler.map(highlight_nan)
    styler.map(highlight_positives, subset = [c.TARGET_COLUMN_NAME])
    styler.set_properties(subset=None, **{"text-align": text_align})
    return styler #update

def calculate_nan_fractions(data: pd.DataFrame,
                            target_column: str = c.TARGET_COLUMN_NAME,
                            ) -> pd.DataFrame:
    """
    Calculates the fraction of the missing values within each column in the dataset (feature).
    
    Parameters
    __________
    data: pd.DataFrame
          Input dataset
    
    target_column: str, optional
          Boolean target column name, by default c.TARGET_COLUMN_NAME

    Returns
    _______
    pd.DataFrame
       Missing value fractions
    """
    #extract columns with null values
    nan_columns = data.columns[data.isnull().any()]
    nan_data = data[nan_columns]
    N = len(nan_data)

    #calculate fractions of null values (all, positive, nagative)
    nan_counts = nan_data.isnull().sum()
    fraction_missing = nan_counts / N
    positives_nan_counts = nan_data[data[target_column]].isnull().sum()
    negatives_nan_counts = nan_data[data[target_column] == False].isnull().sum()
    fraction_missing_positives = positives_nan_counts / nan_counts
    fraction_missing_negatives = negatives_nan_counts / nan_counts

    fraction_missing_df = pd.DataFrame(
        {
        "Total Missing" : fraction_missing,
        "Negatives Fraction" : fraction_missing_negatives,
        "Positives Fraction" : fraction_missing_positives,
        }
    )
    fraction_missing_df.index.name = "Column Name"

    return fraction_missing_df.sort_values("Total Missing", ascending = False)

# helper for cleaning nans
def remove_missing_data_columns(data: pd.DataFrame,
                                threshold: float = c.NAN_FRACTION_THRESHOLD,
                                target_column: str = c.TARGET_COLUMN_NAME
                                ) -> pd.DataFrame:
    """
    Removes columns where the fraction of the missing values is greater than *threshold*.
    
    Parameters
    __________
    data: pd.DataFrame
          Input data
    threshold: float, optional
          Fraction of missing values to use a threshold, by default c.NAN_FRACTION_THRESHOLD
    target_column: str, optional
          Boolean target column name, by default c.TARGET_COLUMN_NAME
    
    Returns
    _______
    pd.Dataframe
       Dataframe without missing values
    """
    missing_fractions = calculate_nan_fractions(data, target_column=target_column)
    flagged = missing_fractions[missing_fractions["Total Missing"] > threshold].index
    return data.drop(flagged, axis = 1)

def clean_missing_values(data: pd.DataFrame,
                         threshold: float = c.NAN_FRACTION_THRESHOLD,
                         target_column: str = c.TARGET_COLUMN_NAME
                        ) -> pd.DataFrame:
    """
    Cleans missing values from the dataset.

    Parameters
        __________
        data: pd.DataFrame
              Input data
        threshold: float, optional
              Fraction of missing values to use a threshold, by default c.NAN_FRACTION_THRESHOLD
        target_column: str, optional
              Boolean target column name, by default c.TARGET_COLUMN_NAME
        
        Returns
        _______
        pd.Dataframe
           NaN-free dataset
    """
    #remove columns with a lot of nans
    data = remove_missing_data_columns(data, threshold=threshold, target_column=target_column)
    #remove observations with nans
    return data.dropna(axis = 0, how = 'any').reset_index(drop = True)