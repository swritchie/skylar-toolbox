# =============================================================================
# Load libraries
# =============================================================================

import networkx as nx
import numpy as np
import pandas as pd
from sklearn import base as snbe
from skylar_toolbox import exploratory_data_analysis as steda
from skylar_toolbox import feature_selection as stfs

# =============================================================================
# CorrelatedDropper
# =============================================================================

class CorrelatedDropper(snbe.BaseEstimator, snbe.TransformerMixin):
    def __init__(self, sample_dt=dict(frac=1e0, random_state=0), threshold_ft=1e0):
        self.sample_dt, self.threshold_ft = sample_dt, threshold_ft
    def fit(self, X, y):
        # Get correlations between features
        self.correlations_between_features_ss = steda.get_correlations(X=X.sample(**self.sample_dt))
        self.correlated_feature_groups_df = get_correlated_groups(correlations_ss=self.correlations_between_features_ss)
        # Get correlations with target
        self.correlations_with_target_ss = steda.get_correlations(X=X.sample(**self.sample_dt), y=y.sample(**self.sample_dt))
        return self
    def transform(self, X): 
        # Get features to drop (high correlation with each other, low correlation with target)
        try: self.drop_lt = stfs.get_correlated_features_to_drop(
            correlated_groups_lt=self.correlated_feature_groups_df.loc[self.threshold_ft, 'groups'], 
            scores_ss=self.correlations_with_target_ss.abs(), 
            lower_is_better_bl=False)
        except: self.drop_lt = []
        return X.drop(columns=self.drop_lt)
    def get_feature_names_out(): pass

# =============================================================================
# get_correlated_groups
# =============================================================================

def get_correlated_groups(correlations_ss, thresholds_ay=np.arange(start=1e-2, stop=1e0, step=1e-2)):
    correlated_groups_dt = {}
    for threshold_ft in thresholds_ay:
        filtered_correlations_df = (correlations_ss
            .abs()
            .pipe(func=lambda x: x[x.ge(other=threshold_ft)])
            .reset_index()
            .set_axis(labels=['source', 'target', 'r'], axis=1))
        gh = nx.from_pandas_edgelist(df=filtered_correlations_df, edge_attr='r')
        correlated_groups_dt[round(number=threshold_ft, ndigits=3)] = list(nx.connected_components(G=gh))
    return (pd.Series(data=correlated_groups_dt, name='groups')
        .drop_duplicates()
        .pipe(func=lambda x: x[x.apply(func=len).gt(other=0)])
        .to_frame()
        .assign(**{
            'n_groups': lambda x: x['groups'].apply(func=len),
            'group_sizes': lambda x: x['groups'].apply(func=lambda x: list(map(len, x))),
            'min_group_sizes': lambda x: x['group_sizes'].apply(func=min),
            'max_group_sizes': lambda x: x['group_sizes'].apply(func=max),
            'total_group_sizes': lambda x: x['group_sizes'].apply(func=sum),
            'n_features_dropped': lambda x: x['total_group_sizes'].sub(other=x['n_groups'])})
        .pipe(func=lambda x: x.set_axis(labels=pd.IntervalIndex.from_breaks(breaks=x.index.tolist() + [1], closed='left'))))
