# =============================================================================
# Load libraries
# =============================================================================

import numpy as np
import pandas as pd

# =============================================================================
# describe
# =============================================================================

def describe(df):
    numeric_description_df = (
        df
        .describe()
        .T
        .assign(
            iqr = lambda x: x['75%'] - x['25%'],
            low_outlier_flag = lambda x: x['min'] < x['25%'] - 1.5 * x['iqr'],
            high_outlier_flag = lambda x: x['max'] > x['75%'] + 1.5 * x['iqr']))
    return pd.concat(objs=[
        df.dtypes.astype(dtype=str).rename(index='dtypes'),
        df.nunique().rename(index='nunique'),
        df.apply(func=lambda x: x.value_counts().nlargest().index.tolist()).rename(index='top_values'),
        df.isna().mean().rename(index='pct_missing'),
        df.isin(values=[-np.inf, np.inf]).sum().rename(index='cnt_inf'),
        numeric_description_df
    ], axis=1).sort_values(by=['dtypes', 'nunique'])

# =============================================================================
# get_differences
# =============================================================================

def get_differences(df, columns_lt): return df.assign(
    abs_diff = lambda x: x.loc[:, columns_lt].diff(axis=1).iloc[:, -1].abs(), 
    mean = lambda x: x.loc[:, columns_lt].mean(axis=1),
    pct_diff = lambda x: x['abs_diff'] / x['mean'])

# =============================================================================
# plot_histogram
# =============================================================================

def plot_histogram(ss, hist_bins_it=int(3e1), table_bbox_lt=[1.25, 0, 2.5e-1, 1]):
    ax = ss.hist(bins=hist_bins_it)
    data_ss = ss.describe().round(decimals=3)
    pd.plotting.table(ax=ax, data=data_ss, bbox=table_bbox_lt)
    return ax

# =============================================================================
# plot_time_series
# =============================================================================

def plot_time_series(datetime_ss, other_ss, resample_rule_sr, table_bbox_lt=[1.25, 0, 2.5e-1, 1]):
    plot_ss = (
        pd.concat(objs=[datetime_ss, other_ss], axis=1)
        .set_index(keys=datetime_ss.name)
        .resample(rule=resample_rule_sr)
        .mean()
        .squeeze())
    data_ss = plot_ss.describe().round(decimals=3)
    ax = plot_ss.plot(marker='.')
    ax.axhline(y=plot_ss.mean(), c='k', ls=':')
    pd.plotting.table(ax=ax, data=data_ss, bbox=table_bbox_lt)
    return ax

# =============================================================================
# plot_value_counts
# =============================================================================

def plot_value_counts(ss, nlargest_n_it=int(1e1), plot_kind_sr='barh', table_bbox_lt=None):
    value_counts_df = (
        ss
        .value_counts(dropna=False)
        .to_frame(name='cnt')
        .assign(pct = lambda x: x['cnt'] / x['cnt'].sum()))
    display_df = value_counts_df.nlargest(n=nlargest_n_it, columns='cnt')
    ax = display_df['cnt'][::-1].plot(kind=plot_kind_sr)
    if table_bbox_lt is None: table_bbox_lt = [1.25, 0, 5e-1, 1e-1 * display_df.shape[0]]
    pd.plotting.table(ax=ax, data=display_df.round(decimals=3), bbox=table_bbox_lt)
    return value_counts_df, ax
