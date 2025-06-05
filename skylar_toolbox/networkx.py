# =============================================================================
# Load libraries
# =============================================================================

import networkx as nx
import numpy as np
import pandas as pd

# =============================================================================
# get_correlated_groups
# =============================================================================

def get_correlated_groups(correlations_ss, thresholds_ay=np.arange(start=1e-2, stop=1e0, step=1e-2)):
    correlated_groups_dt = {}
    for threshold_ft in thresholds_ay:
        filtered_correlations_df = (
            correlations_ss
            .abs()
            .pipe(func=lambda x: x[x.ge(other=threshold_ft)])
            .reset_index()
            .set_axis(labels=['source', 'target', 'r'], axis=1))
        gh = nx.from_pandas_edgelist(df=filtered_correlations_df, edge_attr='r')
        correlated_groups_dt[round(number=threshold_ft, ndigits=3)] = list(nx.connected_components(G=gh))
    return (pd.Series(data=correlated_groups_dt, name='groups')
        .drop_duplicates()
        .to_frame()
        .assign(**{
            'n_groups': lambda x: x['groups'].apply(func=len),
            'group_sizes': lambda x: x['groups'].apply(func=lambda x: list(map(len, x))),
            'min_group_sizes': lambda x: x['group_sizes'].apply(func=min),
            'max_group_sizes': lambda x: x['group_sizes'].apply(func=max),
            'total_group_sizes': lambda x: x['group_sizes'].apply(func=sum),
            'n_features_dropped': lambda x: x['total_group_sizes'].sub(other=x['n_groups'])}))
