# =============================================================================
# Load libraries
# =============================================================================

import itertools
import pandas as pd
import tqdm
from sklearn import base as snbe
from skylar_toolbox import utils as stus

# =============================================================================
# ConstantDropper
# =============================================================================

class ConstantDropper(snbe.BaseEstimator, snbe.TransformerMixin):
    def __init__(self, dropna_bl=False, threshold_ft=1e0): self.dropna_bl, self.threshold_ft = dropna_bl, threshold_ft
    def fit(self, X, y=None):
        self.value_counts_ss = (
            X
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

def get_correlated_features_to_drop(correlated_groups_lt, scores_ss, lower_is_better_bl):
    drop_lt = []
    for correlated_group_st in correlated_groups_lt:
        correlated_scores_ss = scores_ss.loc[list(correlated_group_st)]
        best_sr = correlated_scores_ss.idxmin() if lower_is_better_bl else correlated_scores_ss.idxmax()
        rest_st = correlated_group_st.difference([best_sr])
        drop_lt.extend(list(rest_st))
        stus.print_shapes(sequence=[correlated_group_st, rest_st, drop_lt], sep=' -> ')
    return drop_lt
