# =============================================================================
# Load libraries
# =============================================================================

import itertools
import numpy as np
import pandas as pd
import toolz as tz
import tqdm
from sklearn import base as snbe
from skylar_toolbox import utils as stus

# =============================================================================
# ConstantDropper
# =============================================================================

class ConstantDropper(snbe.BaseEstimator, snbe.TransformerMixin):
    def __init__(self, dropna_bl=False, threshold_ft=1e0): self.dropna_bl, self.threshold_ft = dropna_bl, threshold_ft
    def fit(self, X, y=None):
        self.value_counts_ss = (X
            .apply(func=lambda x: x.value_counts(dropna=self.dropna_bl, normalize=True).iloc[0])
            .sort_values())
        self.constant_ix = self.value_counts_ss.pipe(func=lambda x: x[x.ge(other=self.threshold_ft)]).index
        return self
    def transform(self, X): return X.drop(columns=self.constant_ix)
    def get_feature_names_out(): pass

# =============================================================================
# DuplicatedDropper
# =============================================================================

class DuplicatedDropper(snbe.BaseEstimator, snbe.TransformerMixin):
    def __init__(self): pass
    def fit(self, X, y=None):
        duplicated_lt = []
        for column_sr, column_sr2 in tqdm.tqdm(iterable=sorted(itertools.combinations(iterable=X.columns.sort_values(), r=2))):
            if X[column_sr].equals(other=X[column_sr2]): duplicated_lt.append(column_sr2)
        self.duplicated_ix = pd.Index(data=duplicated_lt).sort_values()
        return self
    def transform(self, X): return X.drop(columns=self.duplicated_ix)
    def get_feature_names_out(): pass

# =============================================================================
# get_correlated_features_to_drop
# =============================================================================

def get_correlated_features_to_drop(correlated_groups_lt, scores_ss, lower_is_better_bl, print_bl=True):
    drop_lt = []
    for correlated_group_st in correlated_groups_lt:
        correlated_scores_ss = scores_ss.loc[list(correlated_group_st)]
        best_sr = correlated_scores_ss.idxmin() if lower_is_better_bl else correlated_scores_ss.idxmax()
        rest_st = correlated_group_st.difference([best_sr])
        drop_lt.extend(list(rest_st))
        if print_bl: stus.print_shapes(x=[correlated_group_st, rest_st, drop_lt], sep=' -> ')
    return drop_lt

# =============================================================================
# get_scores
# =============================================================================

def get_scores(rfecv): return (
    tz.pipe(rfecv.cv_results_, tz.curried.keyfilter(lambda x: x.endswith('score') or x == 'n_features'), pd.DataFrame)
    .set_index(keys='n_features')
    .assign(**{
        'sem_test_score': lambda x: x.filter(regex=r'split\d').sem(axis=1),
        'best': lambda x: x['mean_test_score'].pipe(func=lambda x: x.eq(other=x.max())),
        'best_mean': lambda x: x['mean_test_score'].where(cond=x['best']).bfill().ffill(),
        'best_sem': lambda x: x['sem_test_score'].where(cond=x['best']).bfill().ffill(),
        'best_lower': lambda x: x['best_mean'].sub(other=x['best_sem']),
        'wi_1_sem': lambda x: x['mean_test_score'].gt(other=x['best_lower'])}))

# =============================================================================
# get_support
# =============================================================================

def get_support(rfecv):
    def _get_split_support(support_key_sr): return (
        pd.DataFrame(data=rfecv.cv_results_[support_key_sr], index=rfecv.cv_results_['n_features'])
        .apply(func=np.array, axis=1)
        .apply(func=lambda x: rfecv.feature_names_in_[x]))
    support_keys_lt = tz.pipe(rfecv.cv_results_, tz.curried.keyfilter(lambda x: x.endswith('support')), list)
    return pd.concat(objs=map(_get_split_support, support_keys_lt), axis=1).assign(**{
        'frequencies': lambda x: x.apply(func=lambda x: tz.pipe(x, tz.concat, tz.frequencies), axis=1),
        'intersection': lambda x: x['frequencies'].apply(func=lambda y: list(tz.valfilter(predicate=lambda z: z == x.shape[1] - 1, d=y))),
        'intersection_percentage': lambda x: x['intersection'].apply(func=len).div(other=x.index)})

# =============================================================================
# plot_scores
# =============================================================================

def plot_scores(scores_df):
    n_features_it = scores_df['mean_test_score'].idxmax()
    best_dt = scores_df.loc[[n_features_it], 'mean_test_score'].to_dict()
    title_sr = tz.pipe(best_dt.items(), tz.curried.map(lambda x: 'Features: %d\nScores: %.3f' % x), '\n'.join)
    ax = scores_df.plot(y='mean_test_score', yerr='std_test_score', legend=False, title=title_sr)
    ax.axvline(x=n_features_it, c='k', ls=':')
    return ax
