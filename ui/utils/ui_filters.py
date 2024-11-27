import logging

import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from pandas.core.dtypes.common import is_string_dtype, is_integer_dtype

from utils.config import get_config

logger = logging.getLogger(__name__)

def filter_dataframe(df: pd.DataFrame, df2: pd.DataFrame=None) -> (pd.DataFrame, pd.DataFrame):
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if df is None:
        return None, None
    modify = st.checkbox("Add filters")
    max_rows = get_config('job', 'max_rows')
    max_rows_input = st.number_input(label="max rows", min_value=10, max_value=max_rows, value=max_rows)

    if not modify:
        return (df.head(max_rows_input),
                None if df2 is None else df2.head(max_rows_input))

    df = df.copy()
    if df2 is not None:
        df2 = df2.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            try:
                left, right = st.columns((1, 20))
                left.write("↳")
                if is_integer_dtype(df[column]):
                    _min = int(df[column].min())
                    _max = int(df[column].max())
                    step = 1
                    if _min == _max:
                        pass
                    else:
                        user_num_input = right.slider(
                            f"Values for {column}",
                            _min,
                            _max,
                            (_min, _max),
                            step=step,
                        )
                        df = df[df[column].between(*user_num_input)]
                        if df2 is not None and column in df2:
                            df2 = df2[df2[column].between(*user_num_input)]

                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = right.slider(
                        f"Values for {column}",
                        _min,
                        _max,
                        (_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]
                    if df2 is not None and column in df2:
                        df2 = df2[df2[column].between(*user_num_input)]
                elif is_string_dtype(df[column]):
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]
                    if df2 is not None and column in df2:
                        df2 = df2[df2[column].isin(user_cat_input)]
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len(user_date_input) == 2:
                        user_date_input = tuple(map(pd.to_datetime, user_date_input))
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between(start_date, end_date)]
                        if df2 is not None and column in df2:
                            df2 = df2.loc[df2[column].between(start_date, end_date)]
                else:
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df = df[df[column].str.contains(user_text_input)]
                        if df2 is not None and column in df2:
                            df2 = df2[df2[column].str.contains(user_text_input)]
            except Exception as e:
                logger.info(f"Error filtering column {column}: {e}")
    return (df.head(max_rows_input),
            None if df2 is None else df2.head(max_rows_input))

