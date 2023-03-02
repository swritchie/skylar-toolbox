# =============================================================================
# Load libraries
# =============================================================================

import numpy as np
import pandas as pd

# =============================================================================
# describe_features
# =============================================================================

def describe_features(df: pd.DataFrame):
    '''
    Describes feature matrix

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix

    Returns
    -------
    description_df : pd.DataFrame
        Description

    '''
    numeric_description_df = (
        df
        .describe()
        .T
        .assign(
            oom = lambda x: np.log10(x['50%']).round(), 
            iqr = lambda x: x['75%'] - x['25%'],
            low_outlier_flag = lambda x: x['min'] < x['25%'] - 1.5 * x['iqr'],
            high_outlier_flag = lambda x: x['max'] > x['75%'] + 1.5 * x['iqr']))
    description_df = pd.concat(objs=[
        df.dtypes.astype(dtype=str).rename(index='dtypes'),
        df.nunique().rename(index='nunique'),
        df.apply(func=lambda x: x.value_counts().nlargest().index.tolist()).rename(index='top_values'),
        df.isna().mean().rename(index='pct_missing'),
        numeric_description_df
    ], axis=1)
    description_df.sort_values(by=['dtypes', 'nunique'], inplace=True)
    return description_df

# =============================================================================
# get_differences
# =============================================================================

def get_differences(
        df: pd.DataFrame, 
        columns_lt: list):
    '''
    Gets (percent) differences between pair of columns

    Parameters
    ----------
    df : pd.DataFrame
        Data.
    columns_lt : list
        Columns.

    Returns
    -------
    differences_df : pd.DataFrame
        Data with columns added.

    '''
    differences_df = df.assign(
        abs_diff = lambda x: x.loc[:, columns_lt].diff(axis=1).iloc[:, -1].abs(), 
        mean = lambda x: x.loc[:, columns_lt].mean(axis=1),
        pct_diff = lambda x: x['abs_diff'] / x['mean'])
    return differences_df

# =============================================================================
# get_means
# =============================================================================

def get_means(
        df: pd.DataFrame, 
        columns_lt: list):
    '''
    Gets...
    - mean
    - standard deviation 
    - 2 * standard error of mean
    - lower confidence interval
    - upper confidence interval

    Parameters
    ----------
    df : pd.DataFrame
        Data.
    columns_lt : list
        Columns.

    Returns
    -------
    means_df : pd.DataFrame
        Data with columns added.

    '''
    means_df = df.assign(
        mean = lambda x: x.loc[:, columns_lt].mean(axis=1),
        std = lambda x: x.loc[:, columns_lt].std(axis=1).fillna(value=0),
        se2 = lambda x: 2 * x['std'] / np.sqrt(len(columns_lt)),
        lci = lambda x: x['mean'] - x['se2'],
        uci = lambda x: x['mean'] + x['se2'])
    return means_df

# =============================================================================
# plot_histogram
# =============================================================================

def plot_histogram(
        ss: pd.Series, 
        hist_bins_it: int = 30, 
        table_bbox_lt: list = [1.25, 0, 0.25, 1]):
    '''
    Plots histogram    

    Parameters
    ----------
    ss : pd.Series
        Data to plot
    hist_bins_it : int, optional
        Number of bins for histogram. The default is 30.
    table_bbox_lt : list, optional
        Bounding box for table. The default is [1.25, 0, 0.25, 1].

    Returns
    -------
    ax : plt.Axes
        Axis

    '''
    ax = ss.hist(bins=hist_bins_it)
    data_ss = ss.describe().round(decimals=3)
    pd.plotting.table(ax=ax, data=data_ss, bbox=table_bbox_lt)
    return ax

# =============================================================================
# plot_value_counts
# =============================================================================

def plot_value_counts(
        ss: pd.Series, 
        nlargest_n_it: int = 10, 
        plot_kind_sr: str = 'barh', 
        table_bbox_lt: list = None):
    '''
    Plots and returns value counts    

    Parameters
    ----------
    ss : pd.Series
        Data with values to count
    nlargest_n_it : int, optional
        Number of values to display. The default is 10. All are returned in data frame.
    plot_kind_sr : str, optional
        Argument passed to pd.Series.plot(). The default is 'barh'.
    table_bbox_lt : list, optional
        Bounding box for table. The default is None.

    Returns
    -------
    value_counts_df : pd.DataFrame
        Value counts
    ax : plt.Axes
        Axis

    '''
    value_counts_df = (
        ss
        .value_counts(dropna=False)
        .to_frame(name='cnt')
        .assign(pct = lambda x: x['cnt'] / x['cnt'].sum()))
    display_df = value_counts_df.nlargest(n=nlargest_n_it, columns='cnt')
    ax = display_df['cnt'][::-1].plot(kind=plot_kind_sr)
    if table_bbox_lt is None:
        table_bbox_lt = [1.25, 0, 0.5, 0.1 * display_df.shape[0]]
    pd.plotting.table(ax=ax, data=display_df.round(decimals=3), bbox=table_bbox_lt)
    return value_counts_df, ax
