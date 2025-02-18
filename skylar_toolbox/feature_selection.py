# =============================================================================
# Load libraries
# =============================================================================

import pandas as pd
import tqdm
from sklearn import base as snbe

# =============================================================================
# MulticollinearityMitigator
# =============================================================================

class MulticollinearityMitigator(snbe.BaseEstimator, snbe.TransformerMixin):
    def __init__(self, features_lt, method_sr='pearson', threshold_ft=8e-1): 
        self.features_lt, self.method_sr, self.threshold_ft = features_lt, method_sr, threshold_ft
    def fit(self, X, y):
        # Update features (if previously dropped, etc. in pipeline)
        self.features_lt = X.columns.intersection(other=self.features_lt).tolist()
        # Get target correlations
        target_correlations_dt = {}
        for feature_sr in tqdm.tqdm(iterable=self.features_lt): 
            target_correlations_dt[feature_sr] = abs(X[[feature_sr]].corrwith(other=y, method=self.method_sr).squeeze())
        self.target_correlations_ss = pd.Series(data=target_correlations_dt, name='target_correlations')
        # Get feature groups
        feature_groups_dt = {}
        for feature_sr in tqdm.tqdm(iterable=self.features_lt):
            for feature_sr2 in self.features_lt[self.features_lt.index(feature_sr) + 1:-1]:
                if X[[feature_sr, feature_sr2]].corr(method=self.method_sr).abs().iloc[0, 1] > self.threshold_ft:
                    feature_group_st, feature_group_st2 = map(lambda x: feature_groups_dt.get(x, set()), [feature_sr, feature_sr2])
                    merged_feature_group_st = feature_group_st.union(feature_group_st2, [feature_sr, feature_sr2])
                    for feature_sr3 in merged_feature_group_st: feature_groups_dt[feature_sr3] = merged_feature_group_st
        self.feature_groups_lt = list(set(map(frozenset, feature_groups_dt.values())))
        # Get features to keep and drop
        self.keep_lt, self.drop_lt = [], []
        for feature_group_fs in self.feature_groups_lt:
            self.keep_lt.append(self.target_correlations_ss.loc[feature_group_fs].idxmax())
            self.drop_lt.extend(list(feature_group_fs.difference([self.keep_lt[-1]])))
        return self
    def transform(self, X): return X.drop(columns=self.drop_lt)
