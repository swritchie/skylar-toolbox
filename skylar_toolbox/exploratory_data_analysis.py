# =============================================================================
# Load libraries
# =============================================================================

import itertools
import numpy as np
import pandas as pd
import tqdm

# =============================================================================
# describe
# =============================================================================

def describe(df): return pd.concat(objs=[
    df.dtypes.astype(dtype=str).rename(index='dtypes'),
    df.nunique().rename(index='nunique'),
    df.apply(func=lambda x: x.value_counts(normalize=True).nlargest().round(decimals=3).to_dict()).rename(index='value_counts'),
    df.isin(values=[-np.inf, np.inf]).mean().rename(index='pct_inf'),
    df.isna().mean().rename(index='pct_missing'),
    df.select_dtypes(include='number').lt(other=0).mean().rename(index='pct_negative'),
    df.eq(other=0).mean().rename(index='pct_zero'),
    df.describe().T.drop(columns='count')
    ], axis=1).sort_values(by=['dtypes', 'nunique']).round(decimals=3)

# =============================================================================
# get_correlations
# =============================================================================

def get_correlations(X, y=None):
    method_sr = 'spearman'
    if y is None:
        correlations_dt = {}
        for columns_te in tqdm.tqdm(iterable=sorted(itertools.combinations(iterable=X, r=2))):
            correlations_dt[columns_te] = X[list(columns_te)].corr(method=method_sr).iloc[0, 1]
        return pd.Series(data=correlations_dt, name=method_sr).sort_values()
    else: return X.corrwith(other=y, method=method_sr).rename(index=method_sr).sort_values()

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
# plot_largest_barh 
# =============================================================================

def plot_largest_barh(ss, n_it=int(1e1), signed_bl=True, bbox_lt=[1.25, 0, 2.5e-1, 1]):
    largest_unsigned_ss = ss.abs().nlargest(n=n_it)[::-1]
    largest_signed_ss = ss.loc[largest_unsigned_ss.index].sort_values()
    plot_ss = largest_signed_ss if signed_bl else largest_unsigned_ss
    ax = plot_ss.plot(kind='barh')
    data_ss = plot_ss.reset_index(drop=True).round(decimals=3)[::-1]
    pd.plotting.table(ax=ax, data=data_ss, bbox=bbox_lt)
    return ax

# =============================================================================
# plot_parallel_coordinates
# =============================================================================

def plot_parallel_coordinates(X, y, **kwargs): 
    return pd.plotting.parallel_coordinates(frame=X.join(other=y), class_column=y.name, **kwargs)

# =============================================================================
# plot_radviz
# =============================================================================

def plot_radviz(X, y, **kwargs): 
    return pd.plotting.radviz(frame=X.join(other=y), class_column=y.name, **kwargs)

# =============================================================================
# plot_time_series
# =============================================================================

def plot_time_series(datetime_ss, other_ss, resample_rule_sr, table_bbox_lt=[1.25, 0, 2.5e-1, 1]):
    plot_ss = (pd.concat(objs=[datetime_ss, other_ss], axis=1)
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
    value_counts_df = (ss
        .value_counts(dropna=False)
        .to_frame(name='cnt')
        .assign(pct = lambda x: x['cnt'] / x['cnt'].sum()))
    display_df = value_counts_df.nlargest(n=nlargest_n_it, columns='cnt')
    ax = display_df['cnt'][::-1].plot(kind=plot_kind_sr)
    if table_bbox_lt is None: table_bbox_lt = [1.25, 0, 5e-1, 1e-1 * display_df.shape[0]]
    pd.plotting.table(ax=ax, data=display_df.round(decimals=3), bbox=table_bbox_lt)
    return value_counts_df, ax
